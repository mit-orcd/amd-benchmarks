# Megatron-LM BF16 Benchmark — Result Summary

**Sources**
- `work/log.run` — the wrapping `nohup` driver log (3,294 lines). Tail-end summary printed by `run.sh`.
- `work/logs/bench_bf16_20260528_173512.log` — full in-container training log. (The task referenced `log/…`; the actual directory is `logs/`.)
- Driver: `work/run.sh`, run at `2026-05-28 17:35:12 CDT`.

**Setup recap (from run.sh / log header)**
- 1 node × 8 × AMD Instinct MI355X (gfx950), ROCm 7.2.3, PyTorch 2.8.0a inside `megatron-lm.sif`.
- Workload: GPT, 40 layers, hidden 6144, FFN 16384, 48 heads (GQA, 8 KV groups), seq 4096, SwiGLU + RMSNorm + RoPE, untied embeddings.
- Parallelism: **TP=1, PP=1, DP=8** (pure data-parallel), distributed optimizer ON, `data_parallel_sharding_strategy=no_shard`.
- Batch: micro=2, global=16 (one micro-batch / GPU, no grad-accum).
- Precision: BF16, FlashAttention, mock data, 50 train iters, log every 5.
- Total trainable parameters: **16.22 B** (transformer 15.60 B + embeddings 0.62 B).
- Interconnect: AMD Infinity Fabric / xGMI via RCCL (IB disabled, `NCCL_P2P_DISABLE=0`, `RCCL_MSCCL_ENABLE=1`).

---

## 1. TFLOP/s per GPU  ★

`run.sh` final summary (computed from the 11 `--log-throughput` samples):

```
samples : 11
last    : 239.8 TFLOP/s/GPU
best    : 239.8 TFLOP/s/GPU
```

Per-iteration trace (model-FLOP throughput as reported by Megatron with `--log-throughput`):

| iter | iter time (ms) | TFLOP/s/GPU | notes                                     |
|-----:|---------------:|------------:|-------------------------------------------|
|    1 |       30,072.4 |        27.7 | cold start: RCCL init + kernel autotune   |
|    5 |        9,010.5 |        92.3 | still warming up, hipBLASLt cache filling |
|   10 |        3,656.8 |       227.4 | warm                                      |
|   15 |        3,495.7 |       237.9 | steady                                    |
|   20 |        3,515.3 |       236.6 |                                           |
|   25 |        3,470.4 |       239.6 |                                           |
|   30 |        3,469.9 |       239.7 |                                           |
|   35 |        3,471.2 |       239.6 |                                           |
|   40 |        3,476.2 |       239.2 |                                           |
|   45 |        3,471.2 |       239.6 |                                           |
|   50 |        3,467.4 |       239.8 | best                                      |

**Steady-state: ~239.5 TFLOP/s/GPU, ~3,470 ms/iter, ~1.92 PFLOP/s aggregate over the 8 GPUs.**
Throughput: `16 samples / 3.47 s ≈ 4.6 samples/s ≈ 18.9 k tokens/s` system-wide.

**Model FLOP Utilization (MFU).** MI355X BF16 dense peak ≈ 5 PFLOP/s/GPU → 239.5 / 5000 ≈ **4.8% MFU**. This is low. Suspected drivers (all unchanged by this run):
- PyTorch in the SIF has *no* gfx950 code objects (`torch.cuda.get_arch_list() == []`); the run goes through `HSA_OVERRIDE_GFX_VERSION=9.4.2`, so MI300X (gfx942) kernels execute on MI355X — no gfx950-tuned GEMM/attention.
- FlashAttention 3.0.0.post1 falls outside Transformer Engine's supported window (warning emitted 8×).
- Apex falls back to a *native* RoPE kernel (not the fused one): `UserWarning: Using the native apex kernel for RoPE.`
- Loss is also unusually flat after ~iter 20 (8.12 ± 0.03), consistent with the GQA + cosine schedule converging quickly on mock data; not a perf factor but worth noting.

---

## 2. GPU–GPU communication overhead  ★

Steady-state per-rank timer breakdown at **iter 45** (representative; iter 50 is within ~0.5%):

| stage                                     | time/iter (ms) | % of step | what hits xGMI?                          |
|-------------------------------------------|---------------:|----------:|------------------------------------------|
| forward-compute                           |        ~383    |     11.0  | none (TP=PP=1, local GEMM)               |
| backward-compute                          |       ~2,755   |     79.4  | none (local GEMM)                        |
| forward-backward (sum, reported)          |       3,356.9  |     96.7  | —                                        |
| **all-grads-sync** (DP grad reduce)       |       **196.3**|   **5.66**| **RCCL all-reduce/reduce-scatter (xGMI)**|
| **params-all-gather** (dist-opt)          |        **78.8**|   **2.27**| **RCCL all-gather (xGMI)**               |
| embedding-grads-all-reduce                |          0.01  |    ~0.00  | none in practice — `--untie-embeddings-and-output-weights`, so there is no cross-replica embedding all-reduce |
| optimizer (Adam step + copies)            |          98.5  |     2.84  | local                                    |
| batch-generator                           |           1.25 |     0.04  | local                                    |
| **iter total (reported)**                 |       3,471.2  |    100    | —                                        |

**Cross-GPU communication ≈ 196.3 + 78.8 + 0.01 ≈ 275 ms ≈ 7.9 % of step time.**
The rest of the step is compute-bound. So at this scale, xGMI is **not** the bottleneck — the gap to peak is in single-GPU compute, not comms.

Rank-balance check at iter 45 (max−min across 8 ranks):
- `forward-compute`     spread = 388.5 − 377.2 = **11.3 ms** (~3 % spread)
- `backward-compute`    spread = 2762.4 − 2747.7 = **14.7 ms** (~0.5 %)
- `all-grads-sync`      spread = 196.27 − 196.23 = **0.04 ms**
- `params-all-gather`   spread = 78.90 − 78.78 = **0.12 ms**
- `optimizer`           spread = 98.59 − 98.48 = **0.11 ms**

All 8 GPUs march in lock-step at the collectives → RCCL/MSCCL is using the symmetric xGMI mesh well; no straggler, no NUMA-induced skew despite the NUMA-balancing warning. Tiny spread in forward/backward compute is normal kernel-launch jitter.

**Notes on what is and is **not** stressed.** With TP=PP=1, forward/backward GEMMs stay on-device — the only xGMI traffic is the once-per-step DP gradient reduce and the distributed-optimizer parameter gather. As called out in `notes.md` §“How much is XGMI actually exercised…”, raising TP would amplify xGMI use (per-layer attention/MLP all-reduces); the current run keeps that off the critical path on purpose.

---

## 3. Scaling with number of GPUs  ★

**This run alone is a single 8-GPU point** — there is no 1/2/4-GPU sweep in either log to draw a real scaling curve. What the data *does* say:

1. **Strong rank balance (above).** Every collective is uniform to < 0.2 ms across all 8 ranks; compute spread is < 1 %. That means an 8-way DP step is not bottlenecked by a slow rank, and the xGMI all-to-all topology on this node is being used symmetrically.
2. **Comm fraction is small and grows weakly with DP.** For pure DP + distributed optimizer, comm time scales roughly with `(N-1)/N` of bucket size for reduce-scatter / all-gather. Going from 8→16 on a single fabric domain would change the `(N-1)/N` factor from 0.875 → 0.9375, i.e. ~7 % more comm — still inside the 8 % budget. The cliff is **multi-node**: once a step has to cross InfiniBand instead of xGMI, the 275 ms collective budget will blow up (IB BW is ~10–20× lower than xGMI per link). `run.sh` already disables IB (`NCCL_IB_DISABLE=1`) because this is single-node only.
3. **Per-GPU work is well above the comm/compute crossover.** Each rank owns the full 16 B model (no_shard) and does a 2-sample micro-batch at seq=4096 → ~2.76 s of backward compute vs. ~0.28 s of comm; even doubling N (halving compute per rank via TP or sharding) the ratio stays compute-dominated.
4. **Aggregate scaling estimate at this node size.** From the 8-GPU steady state: ~239.5 TFLOP/s/GPU × 8 = **1.92 PFLOP/s aggregate**. Linearly extrapolating to a perfect-scaling claim is *not* supported by this log — to claim scaling efficiency, re-run at N = 1, 2, 4 (or vary TP/PP) and compare per-GPU TFLOP/s.

**Recommended next experiments to actually quantify scaling**
- Re-run with `N_GPUS = 1, 2, 4, 8` keeping micro-batch fixed → weak-scaling curve & scaling efficiency vs. 8-GPU.
- Sweep `--tensor-model-parallel-size = 1, 2, 4, 8` (with appropriate `--sequence-parallel`) to measure xGMI all-reduce in the hot path; expect comm fraction to climb sharply at TP=4/8.
- Bake a gfx950-native PyTorch into the image and drop `HSA_OVERRIDE_GFX_VERSION` — the 4.8 % MFU is the headline limiter, not communication.

---

## Anomalies & caveats

- **Iter 1 (30 s) and iter 5 (9 s)** dominate the wall time of the first 25 % of the run — exclude them from any throughput average.
- **NCCL/RCCL teardown warnings** at the very end of `log.run` (`Failed to execute operation Close`, `Accept failed Resource temporarily unavailable`) occur *after* iter 50 and the validation/test eval — they are shutdown-time only and do not affect the measured throughput.
- `[aiter] WARNING: NUMA balancing is enabled` — host setting, did not show up as rank skew in this run but should be disabled for production.
- TransformerEngine warns that flash-attn 3.0.0.post1 is outside the supported `[2.1.1, 2.8.0.post2]` range; behavior looked fine here but worth pinning.
- The script enables `--use-distributed-optimizer` while also setting `--data-parallel-sharding-strategy no_shard`. Optimizer-state sharding is still active (hence the ~79 ms `params-all-gather`), but parameter/gradient buffers are full-replicated — which is why per-rank memory sits at 0.64 (a normalized figure) with the full 16 B model.
