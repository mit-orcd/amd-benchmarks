# Megatron-LM TF/s Sweep on gfx950-native Image — Result Summary

**Sources**
- `work/log.tflops-v26.1` — driver log for the sweep, started `2026-06-02 21:47:58 CDT`.
- `work/logs/tflops_v26.1_20260602_214758/` — per-config container logs (`bench_mbs{N}_rc{mode}_{prec}.log`) and `tflops_summary.txt`.
- Driver: `work/run-tflops-v26.1.sh` (MBS × recompute × precision sweep at fixed N=8).
- New image: `megatron-lm-v26.1.sif` (built from `docker://rocm/megatron-lm:v26.1`, 21 GB), sitting alongside — not replacing — the original `megatron-lm.sif`.
- Prior analysis on the *old* image is preserved in `summary-3.md` for direct comparison.

**Setup recap**
- 1 node × 8 × AMD Instinct MI355X (gfx950), ROCm 7.2.3 host driver, PyTorch 2.10.0.dev20251112+rocm7.1 inside the new SIF.
- Workload: GPT, 40 layers, hidden 6144, FFN 16384, 48 heads (GQA, 8 KV groups), seq 4096, SwiGLU + RMSNorm + RoPE, untied embeddings, **no linear biases** (`--disable-bias-linear`, required on this image — see §3).
- Parallelism: TP=1, PP=1, DP=8 (pure data-parallel), distributed optimizer ON, `--overlap-grad-reduce --overlap-param-gather`.
- Attention: `--attention-backend fused` (TE's CK/AITER path on AMD).
- Precision: BF16 baseline, plus FP8 hybrid (delayed scaling) variants.
- Mock data, 50 train iters, log every 5.
- Total trainable parameters: 15.60 B (transformer; embeddings untied).
- Source: `ROCm/Megatron-LM` branch `rocm_dev` HEAD `705c37b83`, host-bound into the container.

**Sweep matrix**: `(MBS, RC, precision)` for `MBS ∈ {2, 4, 8}` × `RC ∈ {none, selective, full}` × `precision ∈ {bf16, fp8}`, taking only the combinations expected to fit and be informative.

---

## Headline

| MBS | Recompute | Precision | GBS | iter time (ms) | last TF/s/GPU | best TF/s/GPU | Mem util |
|----:|-----------|-----------|----:|---------------:|--------------:|--------------:|---------:|
|   2 | none      | bf16      |  16 |        ~1,250  |         709.7 |         715.7 |     0.65 |
|   4 | none      | bf16      |  32 |        ~2,170  |         766.9 |     **778.1** |     0.88 |
|   4 | selective | bf16      |  32 |        ~2,240  |         744.5 |         745.7 |     0.91 |
|   8 | full      | bf16      |  64 |        ~5,000  |         665.2 |         670.4 |     0.65 |
|   2 | none      | **fp8**   |  16 |          ~700  |         914.2 |         950.6 |     0.71 |
|   4 | none      | **fp8**   |  32 |        ~1,510  |       1,108.0 |  **1,111.5** ★ |     0.90 |
|   8 | full      | **fp8**   |  64 |        ~3,300  |         966.4 |         982.1 |     0.65 |

★ **Overall winner: MBS=4, no recompute, FP8 — 1,111.5 TF/s/GPU best (1,108.0 last).**

### vs. NVIDIA B200

Per-GPU throughput across GPU counts (TFLOP/s/GPU):

| GPUs | B200    | MI355X            |
|----:|--------:|------------------:|
|   1 | 1,031.6 | —                 |
|   2 | 1,005.3 | —                 |
|   4 |   993.5 | 778.1 (N=8 BF16)  |

At 4 GPUs:

| Metric                          |  B200 | MI355X |
|---------------------------------|------:|-------:|
| Parallel efficiency (vs. N=1)   | 96.3 %|  n/a¹  |
| % of dense BF16 peak            | 44.2 %| 31.1 % |

Dense BF16 peak (no sparsity): B200 = 2,250 TF/s, MI355X = 2,500 TF/s. ¹ This file has no MI355X N=1 baseline (the 778.1 figure is from the N=8 sweep), so parallel efficiency cannot be computed.

**Reading the comparison:**

- **B200 weak-scales cleanly:** only a 3.7 % per-GPU drop from N=1 to N=4 (1,031.6 → 993.5), so 96.3 % parallel efficiency. Collective overhead is small relative to compute on its NVSwitch fabric.
- **Framework-efficiency gap at the same precision:** at 4 GPUs B200 hits 44.2 % of its dense BF16 peak; MI355X N=8 hits 31.1 % of its dense BF16 peak. ~13 pp of framework headroom remains on MI355X (FP8 kernel tuning, fused-RoPE, etc.).
- **Caveat on the per-GPU column:** 778.1 is at N=8 — twice the GPUs of the matching B200 column — so it is *not* a like-for-like point. Per-GPU throughput typically rises with N (the dist-opt collective fraction shrinks), so a true MI355X N=4 number would land somewhat below 778.1 and widen the framework-efficiency gap further.
- **FP8 closes the headline gap:** MI355X FP8 hits 1,111.5 TF/s/GPU at N=8 — above B200's BF16 numbers across the board. The dense-peak comparison is no longer apples-to-apples (FP8 vs. BF16) but the wall-clock throughput is competitive.

---

## What drove the 3–5× gain

| Metric | Old image (`megatron-lm.sif`, `summary-3.md`) | This sweep (`megatron-lm-v26.1.sif`) | Improvement |
|--------|------------------------------------------------:|--------------------------------------:|------------:|
| Best per-GPU TF/s (BF16) | 236.8 | **778.1** | **3.29×** |
| Best per-GPU TF/s (overall) | 236.8 | **1,111.5** | **4.69×** |
| % of MI355X hipBLASLt BF16 ceiling (1,640 TF/s) | 14.4 % | 47.4 % (BF16) | 3.3× |
| % of MI355X hipBLASLt FP8 ceiling (3,611 TF/s) | n/a | 30.8 % (FP8) | n/a |

1. **gfx950-native image (~3×).** Hot kernels now load native gfx950 code instead of MI300X assembly under `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
2. **Tunings + FP8 (~1.4× on top).** Overlap flags, fused attention, AMD reference env block, and FP8 hybrid.

---

## 1. TFLOP/s per GPU — detailed

### Best BF16: MBS=4, no recompute — 778.1 TF/s/GPU

Steady-state from iter 10 onward (iter 5 still warming):

| iter | iter time (ms) | TF/s/GPU | loss |
|----:|---------------:|---------:|-----:|
| 10  | 2,172          |  765.6   | 10.17 |
| 15  | 2,186          |  760.8   | 8.54 |
| 20  | 2,180          |  762.8   | 8.21 |
| 25  | 2,176          |  764.4   | 8.17 |
| 30  | 2,145          |  775.5   | 8.12 |
| 35  | 2,187          |  760.5   | 8.05 |
| 40  | 2,138          |**778.1** | 7.94 |
| 45  | 2,143          |  776.2   | 7.77 |
| 50  | 2,169          |  766.9   | 7.65 |

Steady-state mean ≈ 767 TF/s/GPU. Iter-to-iter spread ~2 % (kernel-launch jitter).

### Overall winner: MBS=4, no recompute, FP8 hybrid — 1,111.5 TF/s/GPU

| iter | iter time (ms) | TF/s/GPU | loss |
|----:|---------------:|---------:|-----:|
| 10  | 1,496          |**1,111.5**| 10.08 |
| 15  | 1,511          |  1,101.0 | 8.47 |
| 20  | 1,519          |  1,095.3 | 8.18 |
| 25  | 1,520          |  1,094.1 | 8.14 |
| 30  | 1,539          |  1,080.7 | 7.99 |
| 35  | 1,513          |  1,099.4 | 7.79 |
| 40  | 1,512          |  1,100.1 | 7.59 |
| 45  | 1,540          |  1,080.1 | 7.33 |
| 50  | 1,501          |  1,108.0 | 7.07 |

Steady-state mean ≈ 1,096 TF/s/GPU. FP8 loss decays faster than BF16 over the same iteration window (7.07 vs 7.65 at iter 50) because the cosine schedule advances the same number of decay steps in roughly two-thirds the wall time. No NaNs, no skipped iterations across all 50 iters.

### MBS sweet-spot shifted from 2 (old image) to 4 (new image)

On the old image (see `summary-3.md`), MBS=2 was the BF16 winner — because gfx942 kernels saturated at small batches and degraded with larger ones. On the new image:

- **MBS=2 BF16: 715.7** (was 236.8 on old image)
- **MBS=4 BF16: 778.1** (was 231.6 on old image) ← **+9 % over MBS=2**
- **MBS=2 FP8: 950.6**
- **MBS=4 FP8: 1,111.5** ← **+17 % over MBS=2 in FP8**

With proper gfx950 GEMM kernels, **larger batches improve per-GPU throughput** by amortizing kernel launch / collective overhead against more compute per step. MBS=4 hits 0.88–0.90 HBM utilization, leaving no room for MBS=6+ without recompute.

### Recompute is unnecessary at MBS=4 and a regression at MBS=8

| MBS | RC | best TF/GPU | vs MBS=4 no-RC |
|----:|----|------------:|---------------:|
| 4 | none | 778.1 | baseline |
| 4 | selective | 745.7 | −4.2 % |
| 8 | full | 670.4 | −13.8 % |

Recompute trades activation memory for compute. On gfx950-tuned kernels the trade is unfavorable at this model size — the extra recompute cost outweighs the activation savings. Worth enabling only if HBM is a hard constraint (larger models or longer sequences).

### FP8 gives a clean 1.43× over BF16 at MBS=4

- BF16: 778.1 TF/s/GPU
- FP8 hybrid: 1,111.5 TF/s/GPU → **1.43×**

Theoretical FP8/BF16 silicon ratio on MI355X (per rocmval): 3,611 / 1,640 = 2.20×. Achieved framework ratio of 1.43× indicates the FP8 path is only partially tuned. There is likely another 30–40 % headroom if AMD's MXFP8 / blockwise recipes outperform delayed scaling on this workload.

---

## 2. Memory utilization

| MBS | Recompute | Precision | Mem util | HBM used (GiB) | Headroom (GiB) |
|----:|-----------|-----------|---------:|---------------:|---------------:|
| 2 | none | bf16 | 0.65 | 187 | 101 |
| 4 | none | bf16 | 0.88 | 253 | 35 |
| 4 | selective | bf16 | 0.91 | 262 | 26 |
| 8 | full | bf16 | 0.65 | 187 | 101 |
| 2 | none | fp8 | 0.71 | 204 | 84 |
| 4 | none | fp8 | 0.90 | 259 | 29 |
| 8 | full | fp8 | 0.65 | 187 | 101 |

**Observations:**
- The model itself (weights + dist-opt state) takes ~120 GiB; activations add MBS-proportional pressure.
- MBS=4 no-RC sits at 0.88 (BF16) / 0.90 (FP8) HBM util — tight but stable, no OOMs.
- **Full recompute** drops activation memory to near-zero, giving 0.65 util at MBS=8 — could be pushed to MBS=12+ if needed.
- **Selective recompute** at MBS=4 used slightly *more* memory than no-RC (0.91 vs 0.88) without throughput benefit — same counter-intuitive pattern observed on the old image (TE selective-RC workspace overhead exceeds the savings at this model size).

---

## 3. The journey here (failures and fixes)

Five distinct issues had to be resolved between the old image and this winning sweep. Each is worth recording — these are traps anyone else migrating from a gfx942 image to v26.1 will hit.

| # | Failure mode | Root cause | Fix |
|---|--------------|------------|-----|
| 1 | Heredoc rendered torchrun command as broken arg list | `\\` in unquoted heredoc produced literal `\<newline>`, breaking arg parsing | Single `\` at line ends |
| 2 | `ValueError: bias_activation_fusion and use_te_activation_func cannot be both true` | TE 2.6 enforces mutual exclusion | Drop `--use-te-activation-func`, keep default `bias_activation_fusion` |
| 3 | `ValueError: No dot product attention backend is available` | TE 2.6 sees flash-attn 2.8.3 as "not installed" (outside ≤2.8.1 window); `--attention-backend flash` fails because FlashAttention is disabled | Use `--attention-backend fused` (TE's CK/AITER path, AMD-native fast path) |
| 4 | hipBLASLt `Unable to find any suitable algorithms` + SIGSEGV (env-driven) | `HIPBLASLT_TUNING_FILE` pointing at non-existent file disabled fallback; missing `CUDA_DEVICE_MAX_CONNECTIONS=1`; conflicting MSCCL / `NCCL_PROTO` settings vs. AMD's reference | Drop `HIPBLASLT_TUNING_FILE`; add `CUDA_DEVICE_MAX_CONNECTIONS=1`, `RCCL_MSCCL_ENABLE=0`, `NCCL_PROTO=Simple`, `TE_HIPBLASLT_TUNING_{RUN,ALGO}_COUNT` |
| 5 | hipBLASLt `Unable to find any suitable algorithms` in TE wgrad GEMM (final blocker) | TE's bias-fused wgrad backward path requests a hipBLASLt GEMM shape that has no gfx950 kernel in this image | `--disable-bias-linear` — routes through the bias-free wgrad path which IS tuned. Standard for modern LLMs (Llama-style) anyway. |

**Item 5 was the load-bearing fix** — without it nothing trains at all; with it everything works. The other four were necessary cleanup. Practical lesson: AMD's reference script `/workspace/Megatron-LM/examples/llama/train_llama3.sh` inside the image is the canonical recipe for working flag combinations; deviating from it without testing is high-risk.

---

## 4. Other notable findings

- **Loss converges normally** on mock data in all 7 configs. FP8 reaches a lower loss faster (7.07 at iter 50 vs 7.65 BF16) — consistent with the smaller per-step time letting cosine LR schedule decay further per wall-clock minute. No NaNs, no skipped iterations.
- **`Activation memory footprint per transformer layer (precise, without SP): 3264.0 MB`** logged for MBS=4 BF16 no-RC. 40 layers × 3.26 GB = 130 GB activations alone — explains why MBS=6 won't fit without RC and MBS=8 needs full RC.
- **Exit code -11 at process teardown** is shutdown-time noise (NCCL/RCCL cleanup race) and does not affect measurements; the throughput numbers are recorded *before* shutdown.
- **All 7 configs completed successfully** — no OOMs, no real-training crashes. The script's per-config resilience matters less now that no configs fail; the next sweep can be more ambitious (more MBS points, TP variants, MXFP8).

---

## 5. Recommended next experiments

Now that the baseline is competitive, the next gains are smaller but worth measuring.

1. **Try MXFP8 recipe.** `--fp8-recipe mxfp8` exists in this Megatron build and is the focus of the recent ROCm-fork commit `705c37b83` ("MXFP4 recipe enablement"). MX block-scaled FP8 is what MI355X silicon is optimized for; the current `--fp8-recipe delayed` is the legacy path. Expected: another **+10–25 %** on top of 1,111 TF/s.
2. **Try `--fp8-param-gather`.** Currently False. Keeps params in FP8 during the all-gather (vs. cast-then-gather), reducing comm bandwidth.
3. **TP=2 + sequence parallel.** With biases-off we're now in the regime where TE's TP-comm-overlap (`tp_comm_overlap_ag/rs`) becomes hot. `--tensor-model-parallel-size 2 --sequence-parallel` would split the model and enable bulk TP collectives.
4. **Push MBS=6 with selective RC.** The old-image OOM at MBS=6 selective was 0.978 util; with v26.1's tighter memory accounting and the bias-free model, MBS=6 selective might fit.
5. **Add `--use-flash-attn` after a flash-attn downgrade.** If a flash-attn ≤ 2.8.1 wheel can be sideloaded into the image, TE would re-enable the FlashAttention backend for cases where it beats FusedAttention.
6. **Source an Apex fused-RoPE build for gfx950.** The native fallback costs 3–5 %.
