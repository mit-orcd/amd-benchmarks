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

## 4. Comparison to summary-1 (N-GPU sweep baseline)

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

1. **Disable selective recompute and try MBS=5/6 with full recompute.** The selective path is currently broken (uses more memory AND is slower); full recompute at MBS=6–7 should comfortably fit within 288 GiB (MBS=8 full at 0.58 util leaves ~116 GiB headroom).
2. **Push full-recompute MBS higher.** MBS=12 full is at 0.66 util — try MBS=16 (GBS=128) and MBS=20 (GBS=160). The MBS=8 → 12 plateau in throughput (208.0 → 207.2) suggests diminishing returns are already present, but the OOM boundary is not yet located.
3. **Investigate selective recompute compatibility.** The counter-intuitive memory increase at MBS=4 selective vs. none may be a known TE + flash-attn 3.0.0.post1 incompatibility. Upgrading to a TE-compatible flash-attn version (≤ 2.8.0.post2) or gating selective RC through Megatron's native path (not TE) would isolate the cause.
4. **Add TP to the MBS sweep.** With `--tensor-model-parallel-size=2` or `=4`, each rank holds only a shard of the weight matrices, cutting static HBM by 2–4×. This would allow much larger MBS without recompute and exercise the xGMI all-reduce path inside TP.
5. **Bake a gfx950-native PyTorch.** The 4.7 % MFU ceiling is compute-side. Dropping `HSA_OVERRIDE_GFX_VERSION=9.4.2` with a native gfx950 wheel would be the highest-leverage single change for per-GPU throughput.
