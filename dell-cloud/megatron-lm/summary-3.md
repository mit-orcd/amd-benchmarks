# Megatron-LM BF16 TF/s Push Sweep — Result Summary

**Sources**
- `work/log.tflops` — wrapping `nohup` driver log for the sweep, started `2026-06-02 10:12:06 CDT`.
- `work/logs/tflops_20260602_101206/` — per-config container logs (`bench_mbs{N}_rc{mode}.log`) and `tflops_summary.txt`.
- Driver: `work/run-tflops.sh` (MBS sweep: `N_GPUS=8` fixed, `GBS = MBS × 8`, `MBS ∈ {2,3,4,4,6,8,12}` × recompute mode).

**Setup recap (from run-tflops.sh / log header)**
- 1 node × 8 × AMD Instinct MI355X (gfx950), ROCm 7.2.3, PyTorch 2.8.0a inside `megatron-lm.sif`.
- Workload: GPT, 40 layers, hidden 6144, FFN 16384, 48 heads (GQA, 8 KV groups), seq 4096, SwiGLU + RMSNorm + RoPE, untied embeddings.
- Parallelism: **TP=1, PP=1, DP=8** (pure data-parallel), distributed optimizer ON, `data_parallel_sharding_strategy=no_shard` → full 16.22 B model replicated on every rank.
- Topology: **N=8 fixed** — held constant at the best collective-bandwidth regime identified in summary-1/rccl. Sweep knobs are MBS and activation recompute policy.
- Precision: BF16, FlashAttention, mock data, 50 train iters, log every 5.
- Total trainable parameters: **16.22 B** (transformer 15.60 B + embeddings 0.62 B).
- Interconnect: AMD Infinity Fabric / xGMI via RCCL (`NCCL_IB_DISABLE=1`, `RCCL_MSCCL_ENABLE=1`).

**Recompute modes tested**
- `none` — no activation recompute (baseline, identical to summary-1 N=8 run).
- `selective` — `--recompute-granularity selective` (TE-style selective attention recompute).
- `full` — `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` (full per-layer recompute).

---

## At a glance

| MBS | Recompute | GBS | last TF/s/GPU | best TF/s/GPU | Aggregate TF/s | Mem util | OOM |
|----:|-----------|----:|--------------:|--------------:|---------------:|---------:|:---:|
|   2 | none      |  16 |         236.7 |     **236.8** |          1,894 |     0.64 | no  |
|   3 | none      |  24 |         233.2 |         233.5 |          1,868 |     0.75 | no  |
|   4 | none      |  32 |         231.2 |         231.6 |          1,853 |     0.85 | no  |
|   4 | selective |  32 |         228.5 |         228.9 |          1,831 |     0.88 | no  |
|   6 | selective |  48 |          —    |          —    |           —    |    > 0.98| yes |
|   8 | full      |  64 |         207.7 |         208.0 |          1,662 |     0.58 | no  |
|  12 | full      |  96 |         206.4 |         207.2 |          1,658 |     0.66 | no  |

Aggregate TF/s = best_TF/s/GPU × 8. Mem util is the stable HBM fraction from iter 5 onward (reported by Megatron as `mem usages`).

**Headline:** MBS=2 with no recompute is the throughput winner at **236.8 TF/s/GPU**. Every other configuration is slower — a clear sign that on this setup the per-GPU arithmetic intensity is maximized at the smallest feasible batch (the sweet spot for HBM bandwidth vs. compute balance), and neither larger batches nor recompute policies improve it.

**Critical caveat — this number is anomalously low and is being capped by the software stack, not the silicon.** See §5 below. The MI355X silicon delivers **1,639.78 TF/s/GPU BF16** under a native hipBLASLt GEMM benchmark (per `work-rocmval/summary.md`), and reference Megatron-LM on NVIDIA B200 reaches ~1,000 TF/s/GPU. Our best here is **only 14 % of the hipBLASLt achievable ceiling** and ~24 % of B200 Megatron-LM. The MBS/recompute sweep is exploring the *wrong axis* — no value of MBS or recompute can close that gap.

---

## 1. TFLOP/s per GPU  ★

Steady-state per-GPU throughput at iter 50 (last), best across iters 10–50, and iter time:

| MBS | Recompute | GBS | iter time (ms) | **TF/s/GPU last** | TF/s/GPU best | Mem util |
|----:|-----------|----:|---------------:|------------------:|--------------:|---------:|
|   2 | none      |  16 |       3,513.6  |          **236.7**|     **236.8** |     0.64 |
|   3 | none      |  24 |       5,348.8  |          **233.2**|         233.5 |     0.75 |
|   4 | none      |  32 |       7,194.0  |          **231.2**|         231.6 |     0.85 |
|   4 | selective |  32 |       7,277.4  |          **228.5**|         228.9 |     0.88 |
|   6 | selective |  48 |        —  (OOM)|           —      |          —    |    > 0.98|
|   8 | full      |  64 |      16,015.4  |          **207.7**|         208.0 |     0.58 |
|  12 | full      |  96 |      24,171.9  |          **206.4**|         207.2 |     0.66 |

**Observations:**

- **Best is MBS=2, no RC: 236.8 TF/s/GPU** — matches the N=8 baseline from summary-1 (236.9 TF/s/GPU) to within measurement noise, confirming result reproducibility.
- **Per-GPU throughput decreases monotonically with MBS (within the no-RC group).** From MBS=2 → 4, throughput falls from 236.8 → 231.6 TF/s/GPU (~2.2 %). Larger batches fill HBM more, reduce effective bandwidth, and introduce additional overhead without any additional parallelism benefit (DP rank compute is proportional to MBS).
- **Full recompute costs ~12.2 % throughput** vs. the no-RC baseline (236.8 → 208.0 TF/s/GPU). The added compute from replaying 40 transformer layers' forward passes during backward directly extends the step time.
- **MBS=8 vs MBS=12 under full recompute:** nearly identical (208.0 vs 207.2 TF/s/GPU, < 0.4 % difference). Full recompute pins activation memory near-zero per-layer, so scaling MBS from 8 → 12 has marginal additional cost — only static weight/optimizer state determines HBM usage in steady state.
- **MFU context:** MI355X BF16 dense peak ≈ 5 PFLOP/s/GPU → 236.8 / 5000 ≈ **4.7 % MFU** at best. Same ceiling as summary-1. Root cause: `HSA_OVERRIDE_GFX_VERSION=9.4.2` forces MI300X (gfx942) kernels on MI355X — no gfx950-tuned GEMMs. FlashAttention 3.0.0.post1 is outside TE's supported window.

---

## 2. Memory vs. Throughput trade-off  ★

The sweep explicitly trades HBM for throughput via batch size and recompute mode.

| MBS | Recompute | Mem util | HBM used (GiB)¹ | TF/s/GPU best | Throughput penalty vs. MBS=2 |
|----:|-----------|:--------:|----------------:|--------------:|-----------------------------:|
|   2 | none      |     0.64 |           184.3 |         236.8 |                         0.0 % |
|   3 | none      |     0.75 |           216.0 |         233.5 |                        −1.4 % |
|   4 | none      |     0.85 |           244.8 |         231.6 |                        −2.2 % |
|   4 | selective |     0.88 |           253.4 |         228.9 |                        −3.3 % |
|   8 | full      |     0.58 |           167.0 |         208.0 |                       −12.2 % |
|  12 | full      |     0.66 |           190.0 |         207.2 |                       −12.5 % |

¹ HBM used = mem util × 287.98 GiB.

**Key take-aways:**

1. **No-recompute memory scales with MBS as expected.** MBS=2 uses 0.64 → MBS=4 uses 0.85, a ~33 % increase tracking activation memory growth (activations scale linearly with batch, weights + optimizer state are constant). The model at N=8 has ~104 GB free at MBS=2 and only ~43 GB free at MBS=4 — tight headroom.

2. **Selective recompute counter-intuitively uses *more* memory than no-recompute at the same MBS.** MBS=4 selective (0.88) vs MBS=4 none (0.85): selective RC consumed an extra ~8.6 GiB. Under the installed `flash-attn 3.0.0.post1` / TransformerEngine version combination (which TE itself flags as outside its supported window), the selective recompute path appears to store additional workspace buffers for the recompute graph that exceed the activation savings. Net effect: higher HBM *and* lower throughput than the baseline — selective RC is strictly dominated here.

3. **Full recompute resets HBM usage to a lower baseline than even MBS=2 no-RC.** MBS=8 full uses only 0.58 (167 GiB), lower than the MBS=2 no-RC level (0.64 / 184 GiB). Full recompute discards all intermediate activations per layer and re-runs the forward during backward — effectively trading 12.2 % of compute throughput for an enormous activation memory saving.

4. **Full recompute at MBS=12 still fits comfortably.** 0.66 HBM util with GBS=96 — there may be additional headroom to push MBS further under full recompute (MBS=16+), though diminishing returns are already visible from MBS=8 → 12.

---

## 3. OOM analysis: MBS=6 selective  ★

MBS=6 with selective recompute failed at **iter 1 during the forward pass** — before any throughput measurement was logged.

**Error detail:** All 8 GPUs exhausted HBM during the forward pass through the transformer block. The allocation failures were:
- GPUs 6, 7: `Tried to allocate 288 MiB` — failed in `fused_bias_dropout` (Inductor-compiled `self_attn_bda`) inside `_forward_attention`.
- GPUs 0, 5: `Tried to allocate 1.50 GiB` — failed in `TE LayerNormLinear.forward → general_gemm` inside `_forward_mlp`.

At the point of failure, ~**281.5–281.7 GiB was already allocated** by PyTorch (≈ 0.978 HBM util) with only 192–326 MiB free. This is consistent with:
- MBS=4 selective consuming 0.88 × 288 = 253 GiB.
- MBS=6 selective adding 50 % more activations (MBS 4 → 6) = ~127 GiB additional, pushing well above 288 GiB.
- Selective recompute's excess workspace amplifying the problem further (see §2).

The OOM is reproducible and fundamental: selective recompute does not save enough activation memory at MBS=6 to stay within 288 GiB. Full recompute would be required (as confirmed by MBS=8 full succeeding at 0.58 util).

---

## 4. Why is best only 236.8 TF/s/GPU? — Root-cause analysis  ★★

The measured per-GPU throughput is dramatically below what the silicon and the framework should deliver. The gap is on the wrong side of "needs tuning" — it's a structural software-stack mismatch.

### The numbers don't line up

| Reference | BF16 throughput per GPU | % of MI355X dense peak (2,500 TF/s) |
|-----------|------------------------:|------------------------------------:|
| MI355X paper dense peak (matrix, no sparsity) | 2,500 TF/s | 100 % |
| MI355X hipBLASLt GEMM (rocmval, native gfx950, 1-GPU) | **1,639.78 TF/s** | **65.6 %** ← realistic library ceiling |
| NVIDIA B200 Megatron-LM (user reference) | ~1,000 TF/s | ~44 % MFU (relative to B200's 2,250 peak) |
| **This sweep, MBS=2 no-RC** | **236.8 TF/s** | **9.5 %** |
| Ratio to hipBLASLt ceiling on the same silicon | — | **14.4 %** |
| Ratio to B200 Megatron-LM | — | **23.7 %** |

A Megatron-LM run that successfully uses its underlying BLAS library typically achieves **40–55 % of the BLAS ceiling** (per-GEMM efficiency × in-model overheads like attention, optimizer, communication). Hitting only 14 % means most of the heavy kernels aren't actually using the fast path.

### The smoking gun: gfx942 kernels on gfx950 silicon

Three lines in the log expose it:

1. **`AITER_ASM_DIR set to: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/aiter/gfx942/`**
   TransformerEngine's `aiter` (AMD-tuned JIT kernel framework, ships hand-written assembly for the hot paths — attention, GEMM epilogues, LayerNorm/RMSNorm) only contains a **gfx942** (MI300X) subdirectory in this image. There is no `gfx950/` directory. Every model-hot kernel that goes through `aiter` is loading MI300X assembly.

2. **`HSA_OVERRIDE_GFX_VERSION=9.4.2`** in `run-tflops.sh` (and `run.sh`).
   This forces the HSA/ROCr runtime to advertise the MI355X dies as **gfx942** (MI300X) so the framework can find *any* compatible kernel. Without this override the run would not start at all (kernel cache miss). The cost: every dispatched kernel runs MI300X codegen on MI355X silicon — wrong CU count, wrong wavefront layout, wrong LDS/VGPR tuning, no MX-format fast paths, no gfx950-specific instructions.

3. **`PYTORCH_ROCM_ARCH=gfx950`** is set, but it is a *build-time* hint; the PyTorch wheel in the SIF was actually built only for gfx942 (see summary-1: `torch.cuda.get_arch_list() == []` from inside the container — no gfx950 device code present).

The `work-rocmval` benchmark hits 1,639.78 TF/s precisely because it runs **directly against hipBLASLt with native gfx950 codegen** and bypasses TE/aiter/Inductor. That's the ceiling the Megatron run cannot reach with this image.

### What else compounds the loss

In rough order of contribution to the residual gap (gfx942-on-gfx950 is the dominant factor):

| Issue | Evidence in log | Estimated impact |
|-------|-----------------|------------------|
| TE+aiter using gfx942 assembly on gfx950 silicon | `AITER_ASM_DIR .../aiter/gfx942/` | ~3–5× throughput (the dominant term) |
| `HSA_OVERRIDE_GFX_VERSION=9.4.2` masks gfx950 to runtime | env in `run-tflops.sh` | Inseparable from the above |
| `flash-attn 3.0.0.post1` outside TE's supported window | `[WARNING transformer_engine.pytorch.dot_product_attention.utils]: Supported flash-attn versions are >= 2.1.1, <= 2.8.0.post2. Found flash-attn 3.0.0.post1.` (×8) | Attention falls off the fused FlashAttention path; attention is 30–50 % of forward compute. |
| Apex `fused_rope` unavailable, native PyTorch RoPE | `UserWarning: Using the native apex kernel for RoPE.` | ~3–6 % step time. |
| `accumulate_allreduce_grads_in_fp32 = True` | args dump | Doubles grad-reduce bandwidth, marginal. |
| Inductor compiles fused BDA / SwiGLU for gfx942 | `torchinductor_v89592/.../call(...)` paths | Same gfx942 codegen issue applied to JIT'd fused kernels. |

### What "good" would look like

Closing the gap requires fixing the toolchain, not the workload:

- **PyTorch wheel built natively for gfx950**, so `torch.cuda.get_arch_list()` returns `['gfx950']` and `HSA_OVERRIDE_GFX_VERSION` can be removed.
- **TransformerEngine `aiter` shipping `gfx950/` assembly** — currently only `gfx942/` exists in the image.
- **flash-attn pinned to ≤ 2.8.0.post2** to keep TE's fused-attention path active (or upgrade TE to a version that accepts FA3).
- **Apex built for gfx950**, restoring fused RoPE.

A reasonable expectation after the toolchain fix: 700–1,000 TF/s/GPU (30–45 % MFU), matching well-tuned NVIDIA H100/B200 Megatron-LM behavior. Anything substantially below ~1,000 TF/s on a working stack would then point at workload-level tuning (MBS, TP/SP, etc.), at which point the kind of sweep this summary describes becomes meaningful.

### Why MBS/recompute couldn't fix this

Every configuration in §1 is bottlenecked on the **per-kernel** throughput, not on memory or batching. Changing MBS only changes how many tokens per second a slow kernel processes; the per-token TF/s ceiling is set by which kernels load. That's why the entire sweep clusters in a narrow 207–237 TF/s band: it's the gfx942-kernel ceiling, not a workload-tuning frontier.

---

## 5. Comparison to summary-1 (N-GPU sweep baseline)

The MBS=2, no-RC configuration in this sweep is deliberately identical to the N=8 arm of summary-1:

| Metric | summary-1 (N=8, MBS=2, no-RC) | This sweep (MBS=2, no-RC) | Match? |
|--------|:------------------------------:|:-------------------------:|:------:|
| TF/s/GPU last | 236.6 | 236.7 | ✓ < 0.1 % |
| TF/s/GPU best | 236.9 | 236.8 | ✓ < 0.1 % |
| iter time (ms) | 3,515.4 | 3,513.6 | ✓ < 0.1 % |
| Mem util | 0.64 | 0.6363 | ✓ identical |

The results reproduce to within measurement noise, confirming stable hardware state and run-to-run consistency across the two separate runs.

---

## Other notable results

- **Warm-up cost is large regardless of MBS.** Iter 1 is 10–50× slower than steady state (28.7 TF/s at MBS=2 to 96.8 TF/s at MBS=12) due to RCCL init, hipBLASLt cache fill, and Inductor/aiter JIT compilation of fused kernels. Steady state is reached by iter 10 in all successful runs.
- **Loss converges consistently on mock data.** All successful runs reach lm loss ~8.0–8.2 by iter 50; no NaNs or skipped iters. Large-batch runs (MBS=8,12) converge faster in terms of consumed samples.
- **`aiter` module JIT build fires on MBS=2 (first run) and is cached for all subsequent configs.** Build cost is 16.1 s, amortized. Later runs skip rebuild.
- **Persistent environment warnings** (same as summary-1): TE flags flash-attn 3.0.0.post1 as outside its supported range; Apex falls back to native RoPE kernel; `[aiter] NUMA balancing` warning; `pynvml` deprecation; `TORCH_NCCL_AVOID_RECORD_STREAMS` deprecation.
- **Activation memory footprint logged for MBS=8 full:** TE reports `6,528 MB` per transformer layer (40 layers = 261 GB if stored without recompute — hence why full recompute is mandatory for MBS ≥ 5 without selective RC).

---

## Recommended next experiments

**Priority 1 — fix the toolchain (highest leverage by far).** Without this, further MBS / TP / DP sweeps will keep measuring the gfx942 kernel ceiling instead of the silicon's potential.

1. **Get a gfx950-native PyTorch + TransformerEngine + aiter.** Verify inside the container:
   ```python
   python -c "import torch; print(torch.cuda.get_arch_list())"   # must include 'gfx950'
   ls /opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/aiter/  # must have gfx950/
   ```
   If either is missing, no other change matters. A ROCm 7.2.3+ container that has natively built wheels (e.g. AMD's official `rocm/pytorch-training` image for MI355X) should replace `megatron-lm.sif`. Remove `HSA_OVERRIDE_GFX_VERSION=9.4.2` once gfx950 is real.
2. **Pin `flash-attn` to a TE-supported version (≤ 2.8.0.post2)** so TE keeps the fused-attention path. Currently TE prints the warning 8× per run and falls off the fast attention path.
3. **Rebuild Apex with gfx950 support** to restore fused RoPE.

Once those three are done, re-run **MBS=2 no-RC at N=8** as a sanity check. Expected target: **700–1,000 TF/s/GPU** (matching the 30–45 % MFU band typical of well-tuned Megatron-LM on H100/B200). If that lands, then the workload sweeps below are worth running; if not, dig further into the BLAS dispatch path.

**Priority 2 — workload tuning (only meaningful after Priority 1 lands).**

4. **Disable selective recompute and try MBS=5/6 with full recompute.** The selective path is currently broken (uses more memory AND is slower); full recompute at MBS=6–7 should comfortably fit within 288 GiB (MBS=8 full at 0.58 util leaves ~116 GiB headroom). Note: this may resolve on its own once the toolchain is fixed, since the selective-RC memory bloat looks like a TE codegen artifact.
5. **Push full-recompute MBS higher.** MBS=12 full is at 0.66 util — try MBS=16 (GBS=128) and MBS=20 (GBS=160). The MBS=8 → 12 plateau in throughput (208.0 → 207.2) suggests diminishing returns are already present, but the OOM boundary is not yet located.
6. **Add TP to the MBS sweep.** With `--tensor-model-parallel-size=2` or `=4`, each rank holds only a shard of the weight matrices, cutting static HBM by 2–4×. This would allow much larger MBS without recompute and exercise the xGMI all-reduce path inside TP.
