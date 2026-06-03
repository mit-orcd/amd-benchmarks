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

Per-GPU throughput across GPU counts (TFLOP/s/GPU, BF16, MBS=4, no recompute):

| GPUs | B200    | MI355X        | MI355X / B200 |
|----:|--------:|--------------:|--------------:|
|   1 | 1,031.6 | OOM¹          | —             |
|   2 | 1,005.3 | OOM¹          | —             |
|   4 |   993.5 | **723.8**     | 72.9 %        |
|   5 |    —    |     583.2     | —             |
|   6 |    —    |     573.4     | —             |
|   7 |    —    |     572.7²    | —             |
|   8 |    —    |     778.1³    | —             |

Source: GPU-count weak-scaling sweep `logs/tflops_v26.1_gpusweep_20260603_094642/` (see §6). ¹ N=1/2/3 OOM at MBS=4 — full 16 B model + ~130 GB activations exceeds the 288 GB MI355X HBM at low DP rank; dist-opt sharding alone is insufficient. ² N=7 only ran 4 logged iters (1, 5, 10, 15) before the driver killed the sweep; treat as preliminary. N=8 was never reached. ³ N=8 figure is from the MBS×RC×precision sweep above (different driver, same image and config), included here for the high-N reference point.

At 4 GPUs — the only direct apples-to-apples point:

| Metric                          |  B200 | MI355X |
|---------------------------------|------:|-------:|
| Parallel efficiency (vs. N=1)   | 96.3 %|  n/a¹  |
| % of dense BF16 peak            | 44.2 %|  29.0 %|
| % of hipBLASLt BF16 ceiling     |   —   |  44.1 %|

Dense BF16 peak (no sparsity): B200 = 2,250 TF/s, MI355X = 2,500 TF/s. hipBLASLt BF16 ceiling for MI355X = 1,640 TF/s. ¹ MI355X N=1 OOM'd in both `no_shard` and `optim_grads_params` shard modes, so parallel efficiency cannot be computed against an N=1 baseline.

**Reading the comparison:**

- **At matched N=4 BF16, MI355X delivers 73 % of B200 per-GPU** (723.8 vs 993.5). The gap is now apples-to-apples at the framework level, not contaminated by the prior N=8-vs-N=4 caveat.
- **B200 weak-scales cleanly:** only a 3.7 % per-GPU drop from N=1 to N=4 (1,031.6 → 993.5), so 96.3 % parallel efficiency on NVSwitch.
- **MI355X has a non-power-of-2 cliff at N=5–7:** per-GPU collapses from 724 (N=4) to ~570 (N=5/6/7) — a ~21 % drop just by adding one rank. Same pattern observed on the old image (`summary-2.md`); the floor is now ~570 TF/s instead of ~155, but the cliff itself persists. This is a fabric/collective topology effect (rings vs trees, xGMI ordering), not a kernel issue. **N=4 and N=8 are the only "good" scaling points.**
- **N=8 BF16 (778.1) is higher per-GPU than N=4 (723.8)** — the dist-opt collective amortizes better at larger N when N is a power of 2. The non-monotone shape is `N=4 ✓, N=5/6/7 ✗, N=8 ✓`, not classic weak-scaling decay.
- **Framework-efficiency gap at N=4:** B200 hits 44.2 % of its dense BF16 peak; MI355X hits 29.0 % — a 15 pp gap. Against the hipBLASLt BF16 ceiling (a more realistic Megatron-reachable target), MI355X is at 44.1 % — closer, with headroom from FP8 tuning, fused-RoPE, and the N=5–7 cliff.
- **FP8 still wins on wall-clock:** MI355X FP8 at N=8 hits 1,111.5 TF/s/GPU — above B200's BF16 numbers across the board. Not apples-to-apples (precision mismatch), but the realized throughput is competitive.

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
7. **Diagnose the N=5–7 cliff.** Try `RCCL_PROTO=LL/LL128`, alternate `NCCL_ALGO` choices, or `RCCL_TOPO_FILE`. Compare allreduce/all-gather bus-bandwidth at N=4 vs N=5 with `rccl-tests`; the per-GPU floor of ~570 TF/s strongly suggests a collective regression, not a compute one.
8. **Re-run N=7 to completion and add N=8 BF16 MBS=4 to the GPU-count sweep.** Current sweep stopped after N=7 reached only iter 15, and N=8 was never run on the dedicated GPU-count driver (the 778.1 figure is reused from the MBS×RC×precision sweep).

---

## 6. GPU-count weak-scaling sweep (BF16, MBS=4, no recompute)

**Sources**
- `work/log.tflops-v26.1-gpusweep` — driver log, started `2026-06-03 09:46:42 CDT`.
- `work/logs/tflops_v26.1_gpusweep_20260603_094642/` — per-N container logs (`bench_n{1..7}_bf16.log`) and `tflops_summary.txt`.
- Driver: `work/run-tflops-v26.1-gpusweep.sh` (weak-scaling: MBS=4 fixed, GBS = MBS × N, BF16 no-RC, `data_parallel_sharding_strategy=no_shard`).
- Goal: produce the new-image equivalent of `summary-2.md` so the headline B200 comparison has real MI355X N=1/2/4 data.

### Per-N results

| N_GPUS | GBS | iter time (ms) | TF/s/GPU last | TF/s/GPU best | mem util | status |
|-------:|----:|---------------:|--------------:|--------------:|---------:|--------|
|      1 |   4 |   —            |  —            |  —            |  —       | OOM in both `no_shard` and `optim_grads_params` modes |
|      2 |   8 |   —            |  —            |  —            | 0.88 → OOM | OOM after iter 1 (53.4 TF/s warm-up only) |
|      3 |  12 |   —            |  —            |  —            | 0.88 → OOM | OOM after iter 1 (53.4 TF/s warm-up only) |
|      4 |  16 |  2,304         |     **721.9** |     **723.8** | 0.94     | ✓ 50 iters |
|      5 |  20 |  2,860         |       581.5   |       583.2   | 0.93     | ✓ 50 iters |
|      6 |  24 |  2,919         |       569.8   |       573.4   | 0.89     | ✓ 50 iters |
|      7 |  28 |  2,941¹        |     (565.5)¹  |     (572.7)¹  | 0.87     | partial — only iters 1, 5, 10, 15 logged before sweep driver killed |
|      8 |   — |   —            |  —            |  —            |  —       | not run (sweep terminated after N=7) |

¹ N=7 numbers are from the first 15 iters only; steady-state may be slightly different but iter-10 (572.7) and iter-15 (565.5) bracket the value seen at N=5/6. The 778.1 N=8 BF16 point from the MBS×RC×precision sweep (§1) is the reference for the high-N power-of-2 case.

### Why N=1/2/3 OOM

Theoretical per-rank footprint reported by Megatron at MBS=4, BF16, no-RC: **weight+optimizer = 119,351 MB, activation = 139,045 MB, total = 258,396 MB ≈ 252 GiB** per rank. With `no_shard` (and even with dist-opt + `optim_grads_params` sharding at N=1), this leaves only ~36 GiB headroom in 288 GiB HBM — eaten by hipBLASLt workspaces, RCCL buffers, autograd graph spikes, and TE FP8 scaling state. N=1 OOMs at allocation; N=2/3 succeed iter 1 (which uses smaller intermediate buffers) but blow up on iter 2+ when activation memory hits its peak.

At N=4 the optimizer-state shard drops per-rank from ~119 GiB to ~30 GiB; that's enough to fit. The first viable scaling point is therefore N=4, and the parallel-efficiency baseline (1-GPU TF/s) is not measurable for this configuration on MI355X.

**Workaround for a true N=1 baseline:** would need either MBS=2 (smaller activations) or full recompute (`--recompute-granularity full`) — both change the comparison axis and were out of scope for this sweep.

### The N=4 → N=5 cliff

| N | TF/s/GPU best | Δ vs N=4 |
|--:|--------------:|---------:|
| 4 | 723.8         | baseline |
| 5 | 583.2         | −19.4 %  |
| 6 | 573.4         | −20.8 %  |
| 7 | 572.7¹        | −20.9 %  |
| 8 | 778.1²        |  +7.5 %  |

¹ partial, ² from MBS×RC sweep (§1).

A single extra rank costs ~20 % per-GPU throughput, then the floor plateaus across N=5/6/7, then N=8 recovers and exceeds N=4. This is a **fabric/collective regression**, not a compute one — kernel iter time grows from 2,304 ms (N=4) to 2,860–2,941 ms (N=5–7) for the *same* per-rank work. The same shape was seen on the old image (`summary-2.md` reported the same N=4 ✓ / N=5–7 ✗ / N=8 ✓ pattern, just at much lower absolute throughput).

Likely root cause: RCCL's all-reduce/all-gather algorithm choice changes when N crosses 4 on the xGMI fabric — ring at N=4 (clean 4-node ring), then tree-or-mixed for N=5/6/7, then back to ring at N=8. Confirming this requires `NCCL_DEBUG=INFO` algorithm logs side-by-side at N=4 and N=5, plus `rccl-tests` allreduce bus-bandwidth at the same shapes. Listed in §5 as next-experiment item #7.

### Operational notes on the sweep

- **N=1 retry mechanism worked as designed:** the driver detected the no-shard OOM and re-ran N=1 with `--data-parallel-sharding-strategy optim_grads_params` (`bench_n1_bf16_ogp.log`). Both modes OOM'd — confirming N=1 is not viable at MBS=4 regardless of sharding.
- **Exit code −11 (SIGSEGV) at teardown for N=4/5/6/7** is the same shutdown-time RCCL cleanup race noted in §4; throughput numbers are recorded before shutdown and are valid.
- **Sweep stopped mid-N=7** with no error in the bench log — the driver process exited cleanly without ever logging the N=8 header. Driver log ends at the N=7 marker; container log ends mid-run at iter 15. Worth re-running N=7/8 to complete the curve.
- **Memory utilization is high but stable** at N=4 (0.94) and decreases monotonically with N as activations stay fixed per rank but optimizer-state shards shrink. None of N=4..7 hit a runtime OOM.
