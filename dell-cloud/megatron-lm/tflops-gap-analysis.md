# Why 237 TF/s on MI355X vs 1000 TF/s on B200 — and what's tunable in the current script

## 1. Reading the gap as a chain of efficiencies

Per-GPU BF16 throughput, ordered from raw silicon down to the model:

| Layer | MI355X (this setup) | B200 (reference) | What it represents |
|-------|--------------------:|-----------------:|--------------------|
| Paper dense peak (no sparsity) | **2,500 TF/s** | **2,250 TF/s** | Vendor spec, BF16 matrix engine |
| Tuned BLAS (1-GPU GEMM, large matrix) | **1,640 TF/s** (rocmval, gfx950) | ~1,700 TF/s (cuBLASLt, well-tuned) | "Silicon ceiling for a real kernel" |
| % of paper peak hit by tuned BLAS | 66 % | 75 % | Library maturity term |
| **Megatron-LM measured** | **237 TF/s** | **1,000 TF/s** | End-to-end training |
| **Framework efficiency vs. BLAS ceiling** | **14 %** | **59 %** | This is the real comparison |

The right framing isn't "MI355X is 4× slower than B200" — both silicon ceilings are ~the same (1640 vs 1700 TF/s). It's "**MI355X delivers 14 % of its silicon's potential, B200 delivers 59 % of its silicon's potential.**" The ~4× gap is *all* in the framework layer.

### Where the 4× loss lives

Rough decomposition of where MI355X Megatron loses against the hipBLASLt ceiling. These are estimates from the log evidence; they multiply, not add.

| Loss source | Evidence | Estimated impact |
|-------------|----------|------------------|
| **TE / aiter kernels are gfx942 codegen running on gfx950 silicon** | `AITER_ASM_DIR .../aiter/gfx942/`, `HSA_OVERRIDE_GFX_VERSION=9.4.2` | **~2–3×** (dominant — affects every GEMM epilogue, LayerNorm, attention) |
| **FlashAttention 3.0.0.post1 outside TE's supported window** | `Supported flash-attn ... ≤ 2.8.0.post2. Found 3.0.0.post1.` (×8) | **~1.3×** (attention is ~25 % of forward compute, falls off fused path) |
| **Apex `fused_rope` missing → native PyTorch RoPE** | `Using the native apex kernel for RoPE.` | ~1.05× |
| **No DP-collective overlap** | `overlap_grad_reduce=False`, `overlap_param_gather=False` in args dump | ~1.08× (267 ms / 3516 ms exposed at N=8) |
| **BF16 only — FP8 hardware unused** | `fp8 = None` in args dump; MI355X has 2× BF16 throughput in FP8 | **~1.5–2×** (if framework can use it) |
| **No cross-entropy fusion** | `cross_entropy_loss_fusion = False` | ~1.02× |
| **Default attention backend `auto`** | `attention_backend = AttnBackend.auto` | unknown, possibly 1.05–1.1× |

The gfx942-on-gfx950 issue is the elephant — only fixable by replacing the container. **The other items are all script-tunable.**

## 2. What can be improved without changing the container

Listed by expected impact. The ROCm fork at HEAD `705c37b83` (`rocm_dev`) has every flag below — they are not enabled in `run-tflops.sh`.

### Tier A — likely > 30 % improvement, low risk

**A1. Enable DP-collective overlap.** Currently `overlap_grad_reduce=False` and `overlap_param_gather=False`. With `--use-distributed-optimizer` already on, adding:
```
--overlap-grad-reduce
--overlap-param-gather
```
hides the 267 ms grad-reduce + 75 ms param all-gather (summary-1 §2 at N=8) behind backward compute. Expected: **+5–8 % per-GPU TF/s** at N=8, more at larger N.

**A2. Try FP8 training.** MI355X has FP8 matrix-engine peak at 5,000 TF/s (rocmval shows 3,611 TF/s measured — 2.2× BF16). The ROCm fork's recent commit 705c37b83 ("MXFP4 recipe enablement") confirms FP8/MXFP8 paths are exercised in CI. Add:
```
--fp8 hybrid                       # or e4m3
--fp8-recipe delayed               # already the default
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
```
(For MXFP8 specifically, try `--fp8-recipe mxfp8` if TE in the container supports it.) Expected: **+50–100 % per-GPU TF/s** if the FP8 path is healthy; needs a sanity check on numerical convergence. Even an unhealthy path will surface useful diagnostics.

**A3. Force the attention backend.** Currently `auto` — let it pick once with explicit logging:
```
--attention-backend flash          # force flash-attn
```
Then re-run with:
```
--attention-backend fused          # force TE FusedAttention (CK-based on AMD)
```
Whichever wins is the right one. The flash-attn 3.0.0.post1 warning suggests TE's auto-selector is dropping off the fast path; an explicit backend bypasses the version check. Expected: **+5–15 % per-GPU TF/s** if a faster backend exists.

### Tier B — likely 5–15 % improvement

**B1. Cross-entropy loss fusion.**
```
--cross-entropy-loss-fusion
```
Currently `False`. Fuses softmax + log-sum-exp + gather into one kernel. **+2–5 %** depending on vocab size.

**B2. Use TE activation function.** The fork added this in commit `addbda20a`:
```
--use-te-activation-func
```
Routes SwiGLU through TE's fused implementation instead of the PyTorch one. **+3–8 %**.

**B3. Manual GC.** Reduce Python GC pauses on long steps:
```
--manual-gc
--manual-gc-interval 100
```
**+1–3 %**, mostly variance reduction.

**B4. Disable bias-SwiGLU fusion *if* it's hitting a slow path.** Commit `658c3c420` added `--no-bias-swiglu-fusion`. Currently `bias_dropout_fusion=True`. Try toggling — if the fused kernel is the unoptimized Inductor one, the unfused PyTorch path may actually be faster. Measure both.

### Tier C — TP/SP experiments (higher risk, larger upside or regression)

**C1. TP=2 with sequence parallel.**
```
--tensor-model-parallel-size 2
--sequence-parallel
--tp-comm-overlap                  # already wired (tp_comm_bulk_dgrad/wgrad=True)
```
This shrinks per-rank GEMM dimensions but enables TE's TP-comm-overlap path (`tp_comm_overlap_ag/rs` are already True in the dump but inert at TP=1). On MI355X with full xGMI mesh, TP=2 inside a single node is essentially free in bandwidth — the question is whether the smaller GEMM shapes still land on a tuned kernel. Expected: **±15 %**, has to be measured.

**C2. TP=4 with sequence parallel.** Same idea, more aggressive. At N=8 this gives DP=2, TP=4 — well-balanced for the xGMI mesh. **±20 %**, measure.

### Tier D — ROCm environment tuning (low individual impact, additive)

Add to the env block in `run-tflops.sh`:
```bash
TORCH_BLAS_PREFER_HIPBLASLT=1      # force hipBLASLt over legacy hipBLAS
MIOPEN_FIND_MODE=FAST              # or NORMAL; default may re-tune every run
MIOPEN_USER_DB_PATH=/tmp/miopen-$USER
HIPBLASLT_TUNING_FILE=/tmp/hipblaslt-tuning.txt   # captures tuned configs on first run
HIPBLASLT_LOG_MASK=0               # silence logs once tuned
RCCL_MSCCL_ENABLE=1                # already set
NCCL_MIN_NCHANNELS=32              # more channels for small all-reduce
```
Plus drop `NCCL_PROTO=Simple,LL,LL128` → just `LL128` for the smaller per-rank reduce-scatter at TP>1. Expected: **+2–5 %** if MIOpen kernel tuning was firing per-run.

## 3. Sensible test order

Don't enable everything at once. Suggested sequence (each ~5 min at MBS=2 no-RC N=8):

1. **Baseline:** current script, MBS=2 no-RC → 236.7 TF/s (already have this).
2. **+ A1** (overlap-grad-reduce, overlap-param-gather) → expect ~250 TF/s.
3. **+ A3** (explicit `--attention-backend flash`) → expect ~260 TF/s.
4. **+ B1, B2, B3** (fusions + manual GC) → expect ~280 TF/s.
5. **+ A2** (`--fp8 hybrid`) — sanity-check loss convergence over 50 iters → expect **400–500 TF/s** if healthy.
6. **+ C1** (TP=2 SP) at BF16 → measure, may regress or win.
7. **+ C2** (TP=4 SP) at BF16 → measure.
8. **+ Tier D env** → diminishing returns, final polish.

Realistic upper ceiling within the current SIF, with all of the above:

- **BF16 path only:** ~350–450 TF/s/GPU (gfx942 codegen is still the dominant ceiling).
- **FP8/MXFP8 path:** ~600–800 TF/s/GPU (if TE's FP8 dispatch finds working kernels).

Even optimistically, the current container cannot reach 1,000 TF/s without rebuilding to gfx950 native. **But going from 237 → 500+ TF/s is achievable from script changes alone.**

## 4. What would push past the in-script ceiling

If after Tier A+B the number is still well under 500 TF/s, the remaining loss is structural (gfx942 codegen, missing aiter/gfx950, TE+FA version mismatch). At that point the path is replacing `megatron-lm.sif` with a gfx950-native image. See §6 below for sequencing.

## 5. Why B200 hits 1000 TF/s and we expect MI355X (with a fixed stack) to be in the same ballpark

B200's 1000 TF/s at 59 % of its hipBLASLt ceiling is what a fully-tuned NVIDIA stack delivers: cuBLASLt + cuDNN + FlashAttention 3 + TransformerEngine all built natively for sm_100, Apex's fused everything, NCCL collectives, all integration-tested. There's no per-vendor magic — it's years of kernel-library maturity converging on the silicon's actual capability.

The MI355X equivalent — same Megatron source code (already running here), TE-ROCm + AITER + Apex-ROCm + CK-FlashAttention + RCCL all built for gfx950 — has all the same architectural ingredients. AMD's published MI300X MLPerf submissions confirm this works in practice. The fact that this run lands at 14 % of ceiling rather than 50–60 % is specific to **this container image**, not to AMD silicon or ROCm/Megatron-LM the codebase.

## 6. If the goal is best performance: swap the image first

**Yes, and it should be step 1.**

The image swap is worth ~2–3× by itself (gfx942 → gfx950 codegen on every hot kernel — GEMM epilogues, LayerNorm, attention, fused activation). The full Tier A + B + C script work in the current image is worth ~2× at best, and most of those flags need to be enabled on the new image anyway — they are **additive, not alternative**. The hipBLASLt ceiling of 1,640 TF/s is only reachable when kernels are compiled with `gfx950` in `torch.cuda.get_arch_list()` and `transformer_engine/aiter/gfx950/` exists in the image.

A script-first sequence ends at ~500 TF/s against the gfx942 wall, then you swap the image and redo the flag matrix on it anyway. Strictly slower than image-first if the final number is what matters.

### What the new image must have

Five requirements; all five must be satisfied:

1. PyTorch built with `gfx950` in its arch list.
2. TransformerEngine (ROCm) shipping an `aiter/gfx950/` assembly directory.
3. `flash-attn` version inside TE's supported window (≤ 2.8.0.post2 for current TE, or upgrade TE to a version that accepts FA3).
4. Apex (ROCm) with fused-RoPE built for gfx950.
5. RCCL / hipBLASLt / MIOpen current (ROCm 7.2.3+).

Candidate base: `rocm/pytorch-training:vYY.MM` at a tag dated after MI355X / gfx950 GA. By mid-2026 this should exist; verify the tag before committing.

### 30-minute verification before swapping

Run this against the candidate image — it answers all five requirements in one shot:

```bash
singularity exec --rocm <candidate-image>.sif python3 -c "
import torch, transformer_engine, os
print('arch_list :', torch.cuda.get_arch_list())
print('te version:', transformer_engine.__version__)
aiter = os.path.dirname(transformer_engine.__file__) + '/aiter'
print('aiter dirs:', os.listdir(aiter) if os.path.isdir(aiter) else 'no aiter dir')
import flash_attn; print('flash-attn:', flash_attn.__version__)
import apex; print('apex      :', apex.__version__)
"
```

Pass criteria:
- `arch_list` contains `'gfx950'`.
- `aiter dirs` contains `'gfx950'`.
- `flash-attn` version is in TE's supported window.
- `apex` imports cleanly (and ideally `apex.transformer.functional.fused_rope` does not trigger the native-kernel fallback warning at runtime).

If any check fails, that's the gap — either pick a different tag or plan a custom build (hours-to-days, depending on which wheel is missing).

### Recommended sequence (best-performance path)

1. **Verify a candidate image** with the snippet above. If pass → continue. If fail → identify the missing wheel and either find a different tag or build it.
2. **Swap `megatron-lm.sif`** to the verified image. Drop `HSA_OVERRIDE_GFX_VERSION=9.4.2` from `run-tflops.sh` (it's no longer needed).
3. **Reinstall `ROCm/Megatron-LM`** (`rocm_dev` branch, HEAD `705c37b83` or newer) into the image if it isn't already bound from the host.
4. **Rerun MBS=2 no-RC at N=8 as a sanity baseline.** Expected: ~700–900 TF/s/GPU at BF16 with default flags. If this lands below ~500 TF/s, the image is not what it claimed to be — re-verify.
5. **Apply Tier A flags** (overlap-grad-reduce, overlap-param-gather, explicit `--attention-backend flash`). Expected: ~800–1,000 TF/s.
6. **Apply Tier B flags** (cross-entropy fusion, TE activation, manual GC). Expected: ~900–1,050 TF/s.
7. **Enable `--fp8 hybrid`** and verify loss convergence over 50+ iters. Expected: **1,200–1,800 TF/s** if the FP8 path is healthy. (MI355X FP8 hipBLASLt ceiling is 3,611 TF/s; ~40–50 % framework efficiency lands here.)
8. **Tier C TP=2/4 with SP** as a final exploration if HBM headroom allows pushing MBS higher.

Target end state: **1,000+ TF/s/GPU BF16**, **1,500+ TF/s/GPU FP8** — at which point MI355X Megatron is competitive with B200 Megatron on the same workload.

If the image-swap step is blocked (no suitable tag, custom build infeasible right now), the previous script-first path still recovers ~2× and is documented in §2.
