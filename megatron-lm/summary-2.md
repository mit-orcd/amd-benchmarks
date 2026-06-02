# Megatron-LM BF16 Sweep — Result Summary

**Sources**
- `work/log.run-2` — wrapping `nohup` driver log for the sweep, started `2026-05-29 13:11:43 CDT`.
- `work/logs/sweep_20260529_131143/` — per-N container logs (`bench_bf16_n{1..8}.log`) and `sweep_summary.txt`.
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
|      2 |   4 |         202.7 |             405.4 |              85.7 % |                      841 |      20.5 % |
|      3 |   6 |         198.3 |             594.9 |              83.8 % |                      946 |      22.6 % |
|      4 |   8 |         223.3 |             893.2 |              94.4 % |                      486 |      13.1 % |
|      5 |  10 |         156.9 |             784.5 |              66.3 % |                    2,066 |      38.9 % |
|      6 |  12 |         155.7 |             934.2 |              65.8 % |                    2,098 |      39.3 % |
|      7 |  14 |         153.2 |           1,072.4 |              64.8 % |                    2,195 |      40.4 % |
|      8 |  16 |         236.5 |           1,892.0 |             100.0 % |                      272 |       7.7 % |

Parallel efficiency normalized to N=8 (236.5 TF/s/GPU). comm = all-grads-sync + params-all-gather at iter 45.

---

## 1. TFLOP/s per GPU  ★

Steady-state per-GPU throughput at iter 50 (last) and best across iters 10–50:

| N_GPUS | GBS | iter time (ms) | **TF/s/GPU last** | TF/s/GPU best | mem util |
|-------:|----:|---------------:|------------------:|--------------:|---------:|
|      1 |   2 |  —  (OOM)      |  —¹               |  —¹           | 0.99 → OOM |
|      2 |   4 |  4,103.3       | **202.7**         | 202.7         | 0.87 |
|      3 |   6 |  4,194.2       | **198.3**         | 198.4         | 0.75 |
|      4 |   8 |  3,724.0       | **223.3**         | 223.6         | 0.70 |
|      5 |  10 |  5,299.6       | **156.9**         | 157.3         | 0.67 |
|      6 |  12 |  5,340.0       | **155.7**         | 156.0         | 0.64 |
|      7 |  14 |  5,427.5       | **153.2**         | 153.9         | 0.63 |
|      8 |  16 |  3,515.6       | **236.5**         | 237.5         | 0.64 |

¹ **N=1 OOM.** Iter 1 reported 52.6 TF/s/GPU but never reached steady state — `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 287.98 GiB of which 6.00 MiB is free.` The full 16 B model + optimizer state + autograd graph saturates a single 288 GB MI355X (theoretical weight+optimizer = 278 GB, activations = 70 GB, total = 348 GB per the log). With DP ≥ 2 the distributed optimizer shards Adam state and per-rank memory drops from ~0.99 to ≤ 0.87.

**Headline:**
- **Best per-GPU throughput: 237.5 TF/s/GPU at N=8** (powers of 2).
- **N=4 hits 223.6 TF/s/GPU** — close to the N=8 number, so the per-GPU compute kernel itself is fine.
- **N=3, 5, 6, 7 drop to 153–198 TF/s/GPU** — a non-power-of-2 cliff (see §2 & §3).
- **MFU:** MI355X BF16 dense peak ≈ 5 PFLOP/s/GPU → 237.5 / 5000 ≈ **4.7 % MFU** at the best point. Same headline limiter as the prior single-run analysis: PyTorch in the SIF has *no* gfx950 code objects (`torch.cuda.get_arch_list() == []`); the run goes through `HSA_OVERRIDE_GFX_VERSION=9.4.2`, so MI300X (gfx942) kernels execute on MI355X — no gfx950-tuned GEMM/attention. FlashAttention 3.0.0.post1 is also outside Transformer Engine's supported window, and Apex falls back to the *native* RoPE kernel (not the fused one).

---

## 2. GPU–GPU communication overhead  ★

Per-rank timer breakdown averaged across ranks at **iter 45** (steady state; rank-0 numbers shown, others within < 0.2 ms — see balance note below). Forward / backward compute is essentially constant across N because the per-rank batch is constant.

| N | all-grads-sync (ms) | params-all-gather (ms) | optimizer (ms) | fwd-compute (ms) | bwd-compute (ms) | **comm = grads+gather (ms)** | **comm / iter** |
|--:|--------------------:|-----------------------:|---------------:|-----------------:|-----------------:|-----------------------------:|----------------:|
| 2 |               577.9 |                  263.4 |          336.2 |              383 |             2,790 |                    **841**   |     **20.5 %**  |
| 3 |               653.7 |                  292.5 |          340.9 |              382 |             2,792 |                    **946**   |     **22.6 %**  |
| 4 |               334.5 |                  151.3 |          188.4 |              382 |             2,792 |                    **486**   |     **13.1 %**  |
| 5 |             1,359.5 |                  706.4 |          744.4 |              384 |             2,793 |                  **2,066**   |     **38.9 %**  |
| 6 |             1,351.7 |                  746.5 |          775.8 |              386 |             2,793 |                  **2,098**   |     **39.3 %**  |
| 7 |             1,408.9 |                  785.8 |          807.8 |              385 |             2,794 |                  **2,195**   |     **40.4 %**  |
| 8 |               190.3 |                   81.3 |          105.0 |              386 |             2,802 |                    **272**   |      **7.7 %**  |

**Two regimes — power-of-2 vs not.**
- **N ∈ {2, 4, 8}: comm is well-controlled.** N=8 spends only 272 ms (7.7 %) on cross-GPU collectives. N=4 spends 486 ms (13 %). RCCL with `NCCL_ALGO=Ring,Tree` cleanly maps onto the symmetric xGMI mesh, and the distributed-optimizer reduce-scatter / all-gather sees the expected `(N−1)/N` bandwidth term.
- **N ∈ {3, 5, 6, 7}: comm collapses.** Collective time jumps **4–8×** vs. the nearest power-of-2 (N=8 → 272 ms, N=7 → 2,195 ms). This is a classic RCCL ring/tree-topology penalty: non-power-of-2 ring sizes can't form a balanced double-ring on an 8-way xGMI all-to-all mesh, so the all-reduce and all-gather fall back to a slower path. Forward / backward compute is **unchanged** (~385 / ~2,793 ms), confirming the regression is entirely collective-side, not compute-side.
- **Optimizer time tracks all-gather.** `optimizer` (which includes the dist-opt parameter all-gather slice) jumps from 105 ms (N=8) to ~750–810 ms (N=5–7), in lock-step with `params-all-gather`.
- **Embedding-grads-all-reduce ≈ 0.01 ms** everywhere — `--untie-embeddings-and-output-weights` keeps it off the critical path.

**Rank balance (iter 45, max − min spread across all N).**
- `forward-compute`: spread ≤ 9 ms (~2 %), normal kernel-launch jitter.
- `backward-compute`: spread ≤ 16 ms (~0.5 %).
- `all-grads-sync`, `params-all-gather`, `optimizer`: spread < 0.2 ms at every N.

Every collective is uniform across all ranks, including at N=5/6/7 where they are slow. The slowdown is therefore a **collective-algorithm issue**, not a straggler or a NUMA-induced skew, despite the `[aiter] WARNING: NUMA balancing is enabled` notice still appearing in the logs.

---

## 3. Scaling with number of GPUs  ★

This sweep is **weak scaling**: per-GPU micro-batch is fixed (=2), GBS grows with N. Ideal weak-scaling preserves per-GPU TFLOP/s.

Aggregate throughput and weak-scaling efficiency (normalized to the best per-GPU point, N=8 = 236.5 TF/s/GPU):

| N | Aggregate TFLOP/s | Ideal (N × 236.5) | **Weak-scaling efficiency** |
|--:|------------------:|------------------:|----------------------------:|
| 2 |        **405.4**  |             473.0 |                     **85.7 %** |
| 3 |        **594.9**  |             709.5 |                     **83.8 %** |
| 4 |        **893.2**  |             946.0 |                     **94.4 %** |
| 5 |        **784.5**  |           1,182.5 |                     **66.3 %** |
| 6 |        **934.2**  |           1,419.0 |                     **65.8 %** |
| 7 |      **1,072.4**  |           1,655.5 |                     **64.8 %** |
| 8 |      **1,892.0**  |           1,892.0 |                    **100.0 %** |

**Observations.**

1. **Non-monotonic.** Aggregate throughput goes up from N=2 → 4 (almost linearly), **drops** from N=4 → 5 (893.2 → 784.5 TFLOP/s), recovers slowly across N=5,6,7, then **jumps** from N=7 → 8 (1,072 → 1,892). A user picking 5 GPUs to "use what's free" would get *less* aggregate throughput than 4 GPUs.
2. **Powers of 2 dominate.** N=4 (94.4 %) and N=8 (100 %) sit on a clean linear-scaling line; N=3 (83.8 %) is close behind. N=5/6/7 are uniformly ~65 %, capped by the collective overhead identified in §2.
3. **The cliff is RCCL, not xGMI.** Compute per rank is constant (§2). xGMI itself is symmetric — every collective is rank-balanced. The penalty comes from the all-reduce / all-gather algorithm not finding a balanced ring at N ∈ {5, 6, 7}. Likely fixes worth trying *before* re-running: force `NCCL_ALGO=Tree` only, enable `RCCL_MSCCL_ALGO_DIR`, or pin the algorithm via an MSCCL tuning file for those sizes.
4. **Multi-node not exercised.** All comms in this sweep cross **only xGMI** (`NCCL_IB_DISABLE=1`). Once a step crosses InfiniBand, the comm budget would grow ~10–20× per link and the picture changes again.
5. **Single-GPU baseline missing — model is too big for one MI355X under this driver.** Without DP sharding the 16.22 B model + Adam state + activations OOM at 288 GB. To get an N=1 point: enable `--data-parallel-sharding-strategy optim_grads_params` (or activation recompute), or shrink the model.

### Why is N=8 efficiency much higher than the others?

Two effects are mixed in the table above:

**(a) The normalization makes N=8 trivially 100 %.** Aggregate is divided by `N × 236.5` (N=8 per-GPU), so N=8 lands at 100 % by construction. Re-normalized to N=4 (per-GPU = 223.3), N=8 would come out **super-linear at 105.9 %**.

**(b) N=8 really does have the highest per-GPU throughput** (236.5 vs. 223.3 at N=4 vs. 202.7 at N=2). That gap is real and is driven entirely by the collective fraction shrinking with N — two mechanisms, both visible in §2:

- **Distributed-optimizer all-gather data per rank scales as 1/N.** With `--use-distributed-optimizer`, each rank only gathers its shard of the post-step params (G/N bytes per peer). `params-all-gather` goes **263.4 → 151.3 → 81.3 ms** across N=2/4/8 — almost exactly halving each step.
- **Reduce-scatter bandwidth term `(N−1)/N` improves with N.** `all-grads-sync` goes **577.9 → 334.5 → 190.3 ms** across N=2/4/8. The per-rank reduce bucket shrinks, and on a fully-connected xGMI mesh more peers in flight = more concurrent links engaged.

Net effect on the step budget, given per-rank compute is constant (forward ≈ 385 ms, backward ≈ 2,793 ms at every N):

| N | comm fraction | compute fraction | per-GPU TF/s |
|--:|--------------:|-----------------:|-------------:|
| 2 |        20.5 % |           79.5 % |        202.7 |
| 4 |        13.1 % |           86.9 % |        223.3 |
| 8 |         7.7 % |           92.3 % |        236.5 |

Predicted lift N=8 vs. N=4 from compute fraction alone: 0.923 / 0.869 = **1.062 → ~6 %**. Observed: 236.5 / 223.3 = **1.059**. The match is exact — the N=8 advantage is purely the collective fraction shrinking, not a kernel-side speedup.

N=3/5/6/7 fall off this clean line not because the math changes but because RCCL drops onto a slower ring/tree path that can't form a balanced double-ring at those arities (see §2).

---

## Other notable results (not covered above)

- **Memory scales as expected with `--use-distributed-optimizer`.** Per-rank normalized memory drops from 0.87 (N=2) → 0.70 (N=4) → 0.64 (N=8) as the optimizer state shards across DP ranks. `no_shard` is still set for params/grads (full replication), which is why memory plateaus rather than dropping to ~1/N.
- **Iter 1 / iter 5 warm-up dominates the early window.** First-iter times are 7–30 s (RCCL init + hipBLASLt cache fill + Inductor compilation of the SwiGLU fused kernel; e.g. N=8 reports iter 1 = 29.3 s, iter 5 = 8.7 s). Iter 10 onward is steady; exclude iters 1 and 5 from any throughput average.
- **Loss converges quickly to ~7.8–8.2 across all N** on mock data — not a perf factor, but confirms training is numerically progressing (no NaNs, no skipped iters anywhere in the sweep).
- **RCCL teardown warnings at end of each run** (`Failed to execute operation Close`, `Accept failed Resource temporarily unavailable`) are shutdown-time only and do not affect measured throughput.
- **Persistent environment warnings** (carried over from the single-run analysis): TransformerEngine flags flash-attn 3.0.0.post1 as outside its supported window; Apex falls back to a *native* RoPE kernel; `[aiter] WARNING: NUMA balancing is enabled` — host setting, no observed rank-skew effect this sweep.
- **`pynvml` deprecation warning** in `energy_monitor.py` — cosmetic.
- **Run-to-run repeatability is tight.** This sweep was launched ~80 minutes after the prior one ([summary-1.md](summary-1.md)) on the same node with the same image. Per-GPU TF/s lands within ±1.0 of the first sweep at every N (e.g. N=8: 236.6 → 236.5; N=4: 223.7 → 223.3; N=5: 156.8 → 156.9). The non-power-of-2 cliff and the rank-balance picture are identical, so the regression at N=5/6/7 is a deterministic property of the stack, not noise.

---

## Q: Is the non-power-of-2 efficiency drop a hardware-design issue or a Megatron-LM problem?

Neither — it's the collective-communication library (RCCL), not the MI355X hardware and not Megatron-LM.

The evidence is in §2:

- **Per-rank compute is flat across all N.** Forward ~385 ms, backward ~2,793 ms regardless of GPU count. So Megatron-LM's compute path is doing the right thing — nothing in the trainer changes between N=4 and N=5.
- **The entire regression is in `all-grads-sync` + `params-all-gather`.** At N=5/6/7 those jump 4–8× vs. the nearest power of 2 (N=8 → 272 ms total comm, N=7 → 2,195 ms). Everything else is identical.
- **xGMI itself is fine.** Every collective is rank-balanced to <0.2 ms spread across ranks at every N. If it were a hardware topology problem (one link saturated, one die isolated), you'd see straggler ranks. You don't.

So the cliff is **RCCL's algorithm choice** at non-power-of-2 arities. On an 8-way xGMI all-to-all mesh, ring/tree collectives form a clean balanced double-ring at N ∈ {2,4,8}; at N ∈ {5,6,7} they drop onto a fallback path that doesn't pipeline as well. N=3 is interesting — it sits at ~84 % because a 3-ring is small and trivially balanced.

The N=8 number also looks artificially good because of two compounding effects that are *expected*, not bugs: reduce-scatter's `(N−1)/N` bandwidth term and dist-opt's `1/N` per-rank all-gather payload both naturally shrink the comm fraction as N grows. The §3 prediction (1.062×) matches observation (1.059×) almost exactly.

What's worth trying before blaming the stack: rerun N=5/6/7 with `NCCL_ALGO=Tree`, `NCCL_PROTO=Simple`, and `RCCL_MSCCL_ENABLE=0` independently to pin which fallback is firing. If RCCL has MSCCL-tuned algos for 8-way xGMI but not for those arities, that confirms the diagnosis.

---

## Recommended next experiments

1. **Investigate the non-power-of-2 cliff.** Re-run N=5/6/7 with `NCCL_ALGO=Tree` (force tree only), `NCCL_PROTO=Simple` (drop LL/LL128), and `RCCL_MSCCL_ENABLE=0` independently to isolate which fallback path is firing. If MSCCL tuned algorithms exist for AMD 8-way xGMI but not for these arities, that pins the root cause.
2. **Bake a gfx950-native PyTorch.** The 4.7 % MFU ceiling is single-GPU compute, not communication. Drop `HSA_OVERRIDE_GFX_VERSION=9.4.2` once a gfx950 wheel is in the image.
3. **Add a TP/SP sweep at N=8** (`--tensor-model-parallel-size = 1, 2, 4, 8` with `--sequence-parallel`) to put xGMI in the hot path and measure the real all-reduce ceiling.
4. **Enable `optim_grads_params` sharding** so N=1 finishes — needed for a true scaling baseline.
