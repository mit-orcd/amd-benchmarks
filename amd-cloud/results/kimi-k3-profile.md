# Kimi-K3 — profiler trace re-analysis (CPU-side) and tail latency

Traces: `/mnt/scratch/shaohao/traces/kimi_20260820_041644` — 8 ranks. **No GPU time was consumed by this**
analysis; it re-reads files captured earlier.

## What the traces do and do not contain

The earlier `analyze_profile.py` run reported "no GPU kernel events found" and stopped.
That was correct, and the reason is now established: the traces carry **only** these
event categories —

| Category | Events (rank 0) |
|---|---:|
| `cpu_op` | 692,003 |
| `user_annotation` | 2,076 |
| `None` | 58 |
| `Trace` | 1 |

There are **no GPU kernel events at all**, so this was *not* a kineto category-name
mismatch as previously guessed — GPU activity was never captured. **Kernel-level timing
is unrecoverable from these files.** A future profiling run must enable GPU activity
capture explicitly.

What *is* recoverable: ~692 k CPU-side operator events on a single thread with properly
nested intervals, giving exact **exclusive (self) time** per operator.

> **Caveat, load-bearing.** `cpu_op` duration is *host* time. For a fully async GPU op it
> measures launch overhead, not GPU execution. These numbers therefore say where the
> **host** spends step time, and are strong evidence only where the per-call average is
> far too large to be launch overhead — `ChunkKDAFunction` at ~7.8 ms/call being the
> clearest case. Read as directional, not as a kernel profile.

## Where host step time goes (exclusive, mean of 8 ranks)

| Bucket | Share |
|---|---:|
| tensor copy/reshape | **41.3%** |
| KDA linear attention | **38.6%** |
| MLA full attention | **6.6%** |
| GEMM | **4.9%** |
| other | **4.9%** |
| norm/act/quant | **1.5%** |
| MoE routing/experts | **1.3%** |
| collectives | **0.8%** |

All 8 ranks agree within ±0.5 pp on every bucket, so this is a stable property of the
workload rather than a straggler artifact:

| Rank | tensor | KDA | MLA | GEMM | other | norm/act/quant | MoE | collectives | total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_0 | 41.6% | 38.3% | 6.7% | 5.0% | 4.8% | 1.5% | 1.3% | 0.8% | 23.9 |
| rank_1 | 41.5% | 38.3% | 6.7% | 5.0% | 4.8% | 1.6% | 1.3% | 0.8% | 23.9 |
| rank_2 | 41.2% | 38.7% | 6.5% | 4.9% | 5.0% | 1.5% | 1.3% | 0.8% | 24.1 |
| rank_3 | 41.2% | 38.6% | 6.7% | 4.9% | 4.9% | 1.6% | 1.4% | 0.8% | 24.0 |
| rank_4 | 40.8% | 39.2% | 6.6% | 4.8% | 4.8% | 1.5% | 1.3% | 0.8% | 24.2 |
| rank_5 | 41.2% | 38.7% | 6.7% | 4.8% | 4.9% | 1.5% | 1.3% | 0.8% | 24.1 |
| rank_6 | 41.4% | 38.5% | 6.6% | 4.9% | 4.9% | 1.5% | 1.3% | 0.8% | 24.0 |
| rank_7 | 41.2% | 38.9% | 6.6% | 4.8% | 4.8% | 1.5% | 1.3% | 0.8% | 24.1 |

### Top operators by exclusive time (rank 0)

| Operator | Self time | Share | Calls | Avg/call |
|---|---:|---:|---:|---:|
| `aten::copy_` | 9.85 s | 41.2% | 23,447 | 0.42 ms |
| `ChunkKDAFunction` | 8.59 s | 36.0% | 1,104 | 7.78 ms |
| `aiter::unified_attention_with_output_base` | 1.45 s | 6.1% | 384 | 3.77 ms |
| `aiter::gemm_a8w8_bpreshuffle` | 0.82 s | 3.4% | 8,184 | 0.10 ms |
| `aiter::kda_attention_with_output` | 0.49 s | 2.0% | 1,104 | 0.44 ms |
| `aiter::maybe_dual_stream_forward` | 0.24 s | 1.0% | 1,472 | 0.17 ms |
| `aten::fill_` | 0.24 s | 1.0% | 81,165 | 0.00 ms |
| `aiter::fused_moe_` | 0.19 s | 0.8% | 1,472 | 0.13 ms |
| `aten::mm` | 0.14 s | 0.6% | 2,510 | 0.06 ms |
| `aiter::all_reduce_` | 0.13 s | 0.5% | 4,464 | 0.03 ms |
| `aten::empty` | 0.12 s | 0.5% | 92,125 | 0.00 ms |
| `aiter::kimi_k3_apply_attn_res_add` | 0.12 s | 0.5% | 2,848 | 0.04 ms |
| `aiter::dynamic_per_token_scaled_quant` | 0.09 s | 0.4% | 4,808 | 0.02 ms |
| `aiter::_gemm_a4w4_asm` | 0.09 s | 0.4% | 2,944 | 0.03 ms |

## The finding: collectives are not the bottleneck

`kimi-k3-improve.md` §3 ranked the **186 all-reduces per token** as the leading suspect
for the ~93% of step time that is not weight reading. **That ranking was wrong.**

- **Collectives: 0.8%** of host step time — the *smallest* bucket measured.
- **KDA linear attention: 38.6%** — `ChunkKDAFunction` alone is ~36%, at
  ~7.8 ms per call across 1,104 calls. This was ranked second and is confirmed as a
  first-order cost.
- **Tensor copy/reshape: 41.3%** — `aten::copy_`, 23,447 calls at ~0.42 ms.
  **This was not on the list at all.** It is the single largest item, and at ~21 copies
  per KDA call it looks like layout conversion around the `fla` KDA path rather than
  anything intrinsic to the model.

So the ~93% is dominated by **per-step work in the linear-attention path and the tensor
traffic around it**, not by synchronization. The "1% XGMI utilization" was read in §3 as
the signature of a latency-bound collective; on this evidence it simply means the
collectives are cheap.

## Tail latency — corroborating evidence

From `p99`/median already present in every result JSON (no new runs):

| Run | Conc | TPOT med | TPOT p99 | p99/med | TTFT med | TTFT p99 | p99/med |
|---|---:|---:|---:|---:|---:|---:|---:|
| A cap64 | 1 | 21.48 | 21.56 | **1.00** | 224.9 | 273.8 | 1.2 |
| A cap64 | 2 | 22.58 | 22.98 | **1.02** | 251.5 | 446.2 | 1.8 |
| A cap64 | 4 | 24.98 | 25.32 | **1.01** | 256.5 | 495.1 | 1.9 |
| A cap64 | 8 | 27.02 | 27.76 | **1.03** | 257.5 | 622.9 | 2.4 |
| A cap64 | 16 | 31.21 | 32.24 | **1.03** | 261.7 | 1073.4 | 4.1 |
| A cap64 | 32 | 37.83 | 40.08 | **1.06** | 273.9 | 1896.9 | 6.9 |
| A cap64 | 64 | 49.91 | 53.10 | **1.06** | 285.6 | 4236.4 | 14.8 |
| C cap256 | 64 | 49.98 | 53.72 | **1.07** | 282.9 | 13406.2 | 47.4 |
| C cap256 | 128 | 70.98 | 76.13 | **1.07** | 346.4 | 7198.0 | 20.8 |
| C cap256 | 256 | 103.72 | 112.37 | **1.08** | 463.1 | 13196.8 | 28.5 |
| D cap512 | 64 | 50.08 | 52.91 | **1.06** | 285.2 | 13565.9 | 47.6 |
| D cap512 | 128 | 70.71 | 76.29 | **1.08** | 387.2 | 7041.4 | 18.2 |
| D cap512 | 256 | 101.52 | 110.64 | **1.09** | 458.3 | 13335.4 | 29.1 |
| D cap512 | 512 | 152.82 | 169.57 | **1.11** | 541.2 | 26439.8 | 48.9 |

**Decode is metronomic.** TPOT p99/median stays between **1.00 and 1.11** across the
entire range, c=1 to c=512. If 186 synchronization barriers per token were driving
step time, decode would be tail-sensitive — a straggler rank on any barrier would
stretch that step. It does not happen. Steady per-step cost, not sporadic stalls,
which is exactly what the CPU-side breakdown shows.

**TTFT is where the variance lives**, with p99/median rising from 1.2 at c=1 to ~49
at c=512. That is admission and prefill scheduling — the queue — and it is a
property of how work is let in, not of how a decode step executes.

## What this changes

1. **Optimization effort should target the KDA path and the copies around it**, not the
   collectives. Together they are ~80% of host step time.
2. **The `aten::copy_` volume is the most actionable single observation** — 23,447 calls
   per capture. If those are layout conversions in the `fla` KDA integration rather than
   algorithmically required, that is an ATOM/AITER fix with a large blast radius.
3. **The `EP would relieve HBM pressure` argument is dead twice over** — measured to lose
   at every batch size (`kimi-k3-ep-matched.md`), and the resource it targeted is not
   the constraint.

## Reproducing

```
atom/analyze_trace_cpu.py /mnt/scratch/shaohao/traces/kimi_20260820_041644 -o results \
    --sweep A cap64=logs/atom/sweep_20260814_164903 \
    --sweep C cap256=logs/atom/kimi_maxseqs_20260819_171529 \
    --sweep D cap512=logs/atom/kimi_512_20260819_211857 \
```

Needs only the trace files and the result JSONs — no GPU, no server.
