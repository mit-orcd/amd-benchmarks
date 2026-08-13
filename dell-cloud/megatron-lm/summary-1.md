# Megatron-LM BF16 Sweep — Result Summary

**Sources**
- `work/log.run` — wrapping `nohup` driver log for the sweep, started `2026-05-29 11:51:09 CDT`.
- `work/logs/sweep_20260529_115109/` — per-N container logs (`bench_bf16_n{1..8}.log`) and `sweep_summary.txt`.
- Driver: `work/run.sh` (weak-scaling sweep: `MICRO_BS=2` constant, `GBS = MICRO_BS × N_GPUS`, `N_GPUS ∈ {1..8}`).

**Setup recap (from run.sh / log header)**
- 1 node × up to 8 × AMD Instinct MI355X (gfx950), ROCm 7.2.3, PyTorch 2.8.0a inside `megatron-lm.sif`.
- Workload: GPT, 40 layers, hidden 6144, FFN 16384, 48 heads (GQA, 8 KV groups), seq 4096, SwiGLU + RMSNorm + RoPE, untied embeddings.
- Parallelism: **TP=1, PP=1, DP=N** (pure data-parallel), distributed optimizer ON, `data_parallel_sharding_strategy=no_shard` → full 16.22 B model replicated on every rank.
- Batch: micro=2 per GPU, GBS = 2 × N (so per-rank compute per step is *constant* across the sweep — this is weak scaling).
- Precision: BF16, FlashAttention, mock data, 50 train iters, log every 5.
- Total trainable parameters: **16.22 B** (transformer 15.60 B + embeddings 0.62 B).
- Interconnect: AMD Infinity Fabric / xGMI via RCCL (IB disabled, `NCCL_P2P_DISABLE=0`, `RCCL_MSCCL_ENABLE=1`).

---

## At a glance

| N_GPUS | GBS | TF/s/GPU last | Aggregate TFLOP/s | Parallel efficiency | comm = grads+gather (ms) | comm / iter |
|-------:|----:|--------------:|------------------:|--------------------:|-------------------------:|------------:|
|      2 |   4 |         201.2 |             402.4 |              85.0 % |                      841 |      20.4 % |
|      3 |   6 |         197.9 |             593.7 |              83.6 % |                      942 |      22.4 % |
|      4 |   8 |         223.7 |             894.8 |              94.5 % |                      482 |      13.0 % |
|      5 |  10 |         156.8 |             784.0 |              66.3 % |                    2,068 |      39.0 % |
|      6 |  12 |         155.9 |             935.4 |              65.9 % |                    2,105 |      39.4 % |
|      7 |  14 |         153.6 |           1,075.2 |              64.9 % |                    2,190 |      40.4 % |
|      8 |  16 |         236.6 |           1,892.8 |             100.0 % |                      267 |       7.6 % |

Parallel efficiency normalized to N=8 (236.6 TF/s/GPU). comm = all-grads-sync + params-all-gather at iter 45.

---

## 1. TFLOP/s per GPU  ★

Steady-state per-GPU throughput at iter 50 (last) and best across iters 10–50:

| N_GPUS | GBS | iter time (ms) | **TF/s/GPU last** | TF/s/GPU best | mem util |
|-------:|----:|---------------:|------------------:|--------------:|---------:|
|      1 |   2 |  —  (OOM)      |  —¹               |  —¹           | 0.99 → OOM |
|      2 |   4 |  4,132.8       | **201.2**         | 201.8         | 0.87 |
|      3 |   6 |  4,202.8       | **197.9**         | 198.4         | 0.75 |
|      4 |   8 |  3,718.3       | **223.7**         | 223.7         | 0.70 |
|      5 |  10 |  5,304.1       | **156.8**         | 157.2         | 0.67 |
|      6 |  12 |  5,335.6       | **155.9**         | 156.2         | 0.64 |
|      7 |  14 |  5,414.5       | **153.6**         | 153.9         | 0.63 |
|      8 |  16 |  3,515.4       | **236.6**         | 236.9         | 0.64 |

¹ **N=1 OOM.** Iter 1 reported 53.9 TF/s/GPU but never reached steady state — `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 287.98 GiB of which 6.00 MiB is free.` The full 16 B model + optimizer state + autograd graph saturates a single 288 GB MI355X. With DP ≥ 2 the distributed optimizer shards Adam state and per-rank memory drops from ~0.99 to ≤ 0.87.

**Headline:**
- **Best per-GPU throughput: 236.9 TF/s/GPU at N=8** (powers of 2).
- **N=4 hits 223.7 TF/s/GPU** — close to the N=8 number, so the per-GPU compute kernel itself is fine.
- **N=3, 5, 6, 7 drop to 154–198 TF/s/GPU** — a non-power-of-2 cliff (see §2 & §3).
- **MFU:** MI355X BF16 dense peak ≈ 5 PFLOP/s/GPU → 236.9 / 5000 ≈ **4.7 % MFU** at the best point. Same headline limiter as the prior single-run analysis: PyTorch in the SIF has *no* gfx950 code objects (`torch.cuda.get_arch_list() == []`); the run goes through `HSA_OVERRIDE_GFX_VERSION=9.4.2`, so MI300X (gfx942) kernels execute on MI355X — no gfx950-tuned GEMM/attention. FlashAttention 3.0.0.post1 is also outside Transformer Engine's supported window, and Apex falls back to the *native* RoPE kernel (not the fused one).

---

## 2. GPU–GPU communication overhead  ★

Per-rank timer breakdown averaged across ranks at **iter 45** (steady state; rank-0 numbers shown, others within < 0.1 ms — see balance note below). Forward / backward compute is essentially constant across N because the per-rank batch is constant.

| N | all-grads-sync (ms) | params-all-gather (ms) | optimizer (ms) | fwd-compute (ms) | bwd-compute (ms) | **comm = grads+gather (ms)** | **comm / iter** |
|--:|--------------------:|-----------------------:|---------------:|-----------------:|-----------------:|-----------------------------:|----------------:|
| 2 |               574.6 |                  266.8 |          356.3 |              387 |             2,790 |                    **841**   |     **20.4 %**  |
| 3 |               649.4 |                  292.0 |          349.2 |              387 |             2,790 |                    **942**   |     **22.4 %**  |
| 4 |               332.6 |                  149.0 |          188.3 |              383 |             2,790 |                    **482**   |     **13.0 %**  |
| 5 |             1,354.8 |                  712.7 |          753.0 |              384 |             2,790 |                  **2,068**   |     **39.0 %**  |
| 6 |             1,357.5 |                  747.5 |          772.9 |              383 |             2,790 |                  **2,105**   |     **39.4 %**  |
| 7 |             1,422.9 |                  766.8 |          788.7 |              384 |             2,790 |                  **2,190**   |     **40.4 %**  |
| 8 |               191.0 |                   75.5 |           98.5 |              387 |             2,790 |                    **267**   |      **7.6 %**  |

**Two regimes — power-of-2 vs not.**
- **N ∈ {2, 4, 8}: comm is well-controlled.** N=8 spends only 267 ms (7.6 %) on cross-GPU collectives. N=4 spends 482 ms (13 %). RCCL with `NCCL_ALGO=Ring,Tree` cleanly maps onto the symmetric xGMI mesh, and the distributed-optimizer reduce-scatter / all-gather sees the expected `(N−1)/N` bandwidth term.
- **N ∈ {3, 5, 6, 7}: comm collapses.** Collective time jumps **4–7×** vs. the nearest power-of-2 (N=8 → 267 ms, N=7 → 2,190 ms). This is a classic RCCL ring/tree-topology penalty: non-power-of-2 ring sizes can't form a balanced double-ring on an 8-way xGMI all-to-all mesh, so the all-reduce and all-gather fall back to a slower path. Forward / backward compute is **unchanged** (~387 / ~2,790 ms), confirming the regression is entirely collective-side, not compute-side.
- **Optimizer time tracks all-gather.** `optimizer` (which includes the dist-opt parameter all-gather slice) jumps from 98 ms (N=8) to ~750 ms (N=5–7), in lock-step with `params-all-gather`.
- **Embedding-grads-all-reduce ≈ 0.01 ms** everywhere — `--untie-embeddings-and-output-weights` keeps it off the critical path.

**Rank balance (iter 45, max − min spread across all N).**
- `forward-compute`: spread ≤ 8 ms (~2 %), normal kernel-launch jitter.
- `backward-compute`: spread ≤ 14 ms (~0.5 %).
- `all-grads-sync`, `params-all-gather`, `optimizer`: spread < 0.2 ms at every N.

Every collective is uniform across all ranks, including at N=5/6/7 where they are slow. The slowdown is therefore a **collective-algorithm issue**, not a straggler or a NUMA-induced skew, despite the `[aiter] WARNING: NUMA balancing is enabled` notice still appearing in the logs.

---

## 3. Scaling with number of GPUs  ★

This sweep is **weak scaling**: per-GPU micro-batch is fixed (=2), GBS grows with N. Ideal weak-scaling preserves per-GPU TFLOP/s.

Aggregate throughput and weak-scaling efficiency (normalized to the best per-GPU point, N=8 = 236.6 TF/s/GPU):

| N | Aggregate TFLOP/s | Ideal (N × 236.6) | **Weak-scaling efficiency** |
|--:|------------------:|------------------:|----------------------------:|
| 2 |        **402.4**  |             473.2 |                     **85.0 %** |
| 3 |        **593.7**  |             709.8 |                     **83.6 %** |
| 4 |        **894.8**  |             946.4 |                     **94.5 %** |
| 5 |        **784.0**  |           1,183.0 |                     **66.3 %** |
| 6 |        **935.4**  |           1,419.6 |                     **65.9 %** |
| 7 |      **1,075.2**  |           1,656.2 |                     **64.9 %** |
| 8 |      **1,892.8**  |           1,892.8 |                    **100.0 %** |

**Observations.**

1. **Non-monotonic.** Aggregate throughput goes up from N=2 → 4 (almost linearly), **drops** from N=4 → 5 (894.8 → 784.0 PFLOP/s), recovers slowly across N=5,6,7, then **jumps** from N=7 → 8 (1,075 → 1,893). A user picking 5 GPUs to "use what's free" would get *less* aggregate throughput than 4 GPUs.
2. **Powers of 2 dominate.** N=4 (94.5 %) and N=8 (100 %) sit on a clean linear-scaling line; N=3 (83.6 %) is close behind. N=5/6/7 are uniformly ~65 %, capped by the collective overhead identified in §2.
3. **The cliff is RCCL, not xGMI.** Compute per rank is constant (§2). xGMI itself is symmetric — every collective is rank-balanced. The penalty comes from the all-reduce / all-gather algorithm not finding a balanced ring at N ∈ {5, 6, 7}. Likely fixes worth trying *before* re-running: force `NCCL_ALGO=Tree` only, enable `RCCL_MSCCL_ALGO_DIR`, or pin the algorithm via an MSCCL tuning file for those sizes.
4. **Multi-node not exercised.** All comms in this sweep cross **only xGMI** (`NCCL_IB_DISABLE=1`). Once a step crosses InfiniBand, the comm budget would grow ~10–20× per link and the picture changes again.
5. **Single-GPU baseline missing — model is too big for one MI355X under this driver.** Without DP sharding the 16.22 B model + Adam state + activations OOM at 288 GB. To get an N=1 point: enable `--data-parallel-sharding-strategy optim_grads_params` (or activation recompute), or shrink the model.

### Why is N=8 efficiency much higher than the others?

Two effects are mixed in the table above:

**(a) The normalization makes N=8 trivially 100 %.** Aggregate is divided by `N × 236.6` (N=8 per-GPU), so N=8 lands at 100 % by construction. Re-normalized to N=4 (per-GPU = 223.7), N=8 would come out **super-linear at 105.8 %**.

**(b) N=8 really does have the highest per-GPU throughput** (236.6 vs. 223.7 at N=4 vs. 201.2 at N=2). That gap is real and is driven entirely by the collective fraction shrinking with N — two mechanisms, both visible in §2:

- **Distributed-optimizer all-gather data per rank scales as 1/N.** With `--use-distributed-optimizer`, each rank only gathers its shard of the post-step params (G/N bytes per peer). `params-all-gather` goes **266.8 → 149.0 → 75.5 ms** across N=2/4/8 — almost exactly halving each step.
- **Reduce-scatter bandwidth term `(N−1)/N` improves with N.** `all-grads-sync` goes **574.6 → 332.6 → 191.0 ms** across N=2/4/8. The per-rank reduce bucket shrinks, and on a fully-connected xGMI mesh more peers in flight = more concurrent links engaged.

Net effect on the step budget, given per-rank compute is constant (forward ≈ 387 ms, backward ≈ 2,790 ms at every N):

| N | comm fraction | compute fraction | per-GPU TF/s |
|--:|--------------:|-----------------:|-------------:|
| 2 |        20.4 % |           79.6 % |        201.2 |
| 4 |        13.0 % |           87.0 % |        223.7 |
| 8 |         7.6 % |           92.4 % |        236.6 |

Predicted lift N=8 vs. N=4 from compute fraction alone: 0.924 / 0.870 = **1.062 → ~6 %**. Observed: 236.6 / 223.7 = **1.058**. The match is exact — the N=8 advantage is purely the collective fraction shrinking, not a kernel-side speedup.

N=3/5/6/7 fall off this clean line not because the math changes but because RCCL drops onto a slower ring/tree path that can't form a balanced double-ring at those arities (see §2).

---

## Other notable results (not covered above)

- **Memory scales as expected with `--use-distributed-optimizer`.** Per-rank normalized memory drops from 0.87 (N=2) → 0.70 (N=4) → 0.64 (N=8) as the optimizer state shards across DP ranks. `no_shard` is still set for params/grads (full replication), which is why memory plateaus rather than dropping to ~1/N.
- **Iter 1 / iter 5 warm-up dominates the early window.** First-iter times are 7–30 s (RCCL init + hipBLASLt cache fill + Inductor compilation of the SwiGLU fused kernel). Iter 10 onward is steady; exclude iters 1 and 5 from any throughput average.
- **Loss converges quickly to ~8.0–8.3 across all N** on mock data — not a perf factor, but confirms training is numerically progressing (no NaNs, no skipped iters anywhere in the sweep).
- **RCCL teardown warnings at end of each run** (`Failed to execute operation Close`, `Accept failed Resource temporarily unavailable`) are shutdown-time only and do not affect measured throughput.
- **Persistent environment warnings** (carried over from the single-run analysis): TransformerEngine flags flash-attn 3.0.0.post1 as outside its supported window; Apex falls back to a *native* RoPE kernel; `[aiter] WARNING: NUMA balancing is enabled` — host setting, no observed rank-skew effect this sweep.
- **`pynvml` deprecation warning** in `energy_monitor.py` — cosmetic.

---

## Recommended next experiments

1. **Investigate the non-power-of-2 cliff.** Re-run N=5/6/7 with `NCCL_ALGO=Tree` (force tree only), `NCCL_PROTO=Simple` (drop LL/LL128), and `RCCL_MSCCL_ENABLE=0` independently to isolate which fallback path is firing. If MSCCL tuned algorithms exist for AMD 8-way xGMI but not for these arities, that pins the root cause.
2. **Bake a gfx950-native PyTorch.** The 4.7 % MFU ceiling is single-GPU compute, not communication. Drop `HSA_OVERRIDE_GFX_VERSION=9.4.2` once a gfx950 wheel is in the image.
3. **Add a TP/SP sweep at N=8** (`--tensor-model-parallel-size = 1, 2, 4, 8` with `--sequence-parallel`) to put xGMI in the hot path and measure the real all-reduce ceiling.
4. **Enable `optim_grads_params` sharding** so N=1 finishes — needed for a true scaling baseline.
