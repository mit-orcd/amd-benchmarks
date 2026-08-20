# Kimi-K3 — single-stream (per-request) speed vs kernel path

Source run: `kimi_single_stream_20260820_174733`. Low-concurrency sweep across kernel-path configurations,
all on the MAD-pinned image, TP=8, ISL/OSL 1024/1024.

**Primary metric is per-request tok/s (`1000 / median TPOT`)** — the decode rate one user
experiences — not aggregate throughput. This is the only experiment in the Kimi-K3 set
that targets it; every other one measured aggregate tok/s, where batching dominates.

## Arms

| Arm | Change from MAD baseline |
|---|---|
| `K1_mad_default` | MAD baseline (control) |
| `K3_aiter_attn` | `ATOM_USE_UNIFIED_ATTN=0`, `ATOM_FORCE_ATTN_TRITON=0` — attention path |
| `K4_grouped_gemm` | `ATOM_USE_TRITON_GEMM=0`, `AITER_USE_GROUPED_GEMM=1` — GEMM path |

Each arm flips exactly one kernel-path decision, so a difference is attributable.

## Per-request tok/s (higher is better)

| Concurrency | `K1_mad_default` | `K3_aiter_attn` | `K4_grouped_gemm` |
|---:|---:|---:|---:|
| 1 | 40.9 | 40.8 | 42.1 |
| 2 | 39.7 | 39.8 | 40.8 |
| 4 | 37.7 | 37.7 | 38.7 |
| 8 | 34.7 | 34.8 | 35.5 |

## Median TPOT (ms, lower is better)

| Concurrency | `K1_mad_default` | `K3_aiter_attn` | `K4_grouped_gemm` |
|---:|---:|---:|---:|
| 1 | 24.44 | 24.49 | 23.78 |
| 2 | 25.21 | 25.14 | 24.48 |
| 4 | 26.53 | 26.49 | 25.85 |
| 8 | 28.81 | 28.73 | 28.19 |

## Aggregate tok/s (secondary)

| Concurrency | `K1_mad_default` | `K3_aiter_attn` | `K4_grouped_gemm` |
|---:|---:|---:|---:|
| 1 | 40.5 | 40.5 | 41.7 |
| 2 | 77.2 | 77.4 | 79.4 |
| 4 | 144.2 | 144.6 | 148.2 |
| 8 | 231.7 | 231.8 | 235.9 |

## Reading

Control `K1_mad_default` at c=1: **40.9 tok/s per request** (TPOT 24.44 ms).

| Arm | per-request tok/s @ c=1 | vs control |
|---|---:|---:|
| `K3_aiter_attn` | 40.8 | -0.2% |
| `K4_grouped_gemm` | 42.1 | +2.8% |

**No kernel path meaningfully changes single-stream speed** — the full spread is within ±5% of the control. This is a clean negative result and it is informative: it localizes the per-request cost to things no environment variable can reach — kernel launch overhead, the 186 serialized all-reduces (fixed by TP=8 × 93 layers), and the sequential dependency in the 69 KDA layers. **Configuration-level tuning for per-request speed is closed**; further gains require ATOM/AITER changes.

**Context — Run A measured 46.6 tok/s at c=1** (TPOT 21.48 ms) on `rocm/atom-dev:latest`. That is a *different image*, so it is not a clean comparison against these arms; `K1_mad_default` is the matched control. The two are listed together only to show the regime is consistent.

## The headroom this was testing against

At c=1 a decode step reads ~3.4 GB of weights per GPU. At the ~2.2 TB/s effective rate
implied by the c=64 measurement (116 GB / 51.7 ms), that is **~1.5 ms** of weight
reading against a measured TPOT of ~21 ms — so **~93% of a single-request step is not
weight traffic**. That 93% is serialization, and this experiment asks how much of it is
reachable from configuration. Whatever the answer, the ~1.5 ms figure is a floor and
not a target: kernel dispatch, the KDA dependency chain, and a real minimum collective
latency are all irreducible.

Levers deliberately **not** tested here, because they are unavailable rather than
untried: reducing the 186 all-reduces (fixed by TP=8 × 93 layers, no flag); TP<8 with
replicas (1.5 TB of weights against 2.3 TB of HBM forces TP=8 on one node);
speculative decoding / MTP (`num_nextn_predict_layers = 0` — no MTP heads shipped);
HIP graphs (already enabled, server log reports `cudagraph=True`).

## Source data

| What | Where |
|---|---|
| Per-arm JSON / logs | `kimi_single_stream_20260820_174733/<arm>/c<N>.{json,log}` |
| Per-arm env (exact vars) | `kimi_single_stream_20260820_174733/<arm>/env.txt` |
| Server logs | `kimi_single_stream_20260820_174733/<arm>/atom_server.log` |
| Driver state | `kimi_single_stream_20260820_174733/STATE.txt` |
| This data as CSV | `results/kimi-k3-single-stream.csv` |
| Context and rationale | `kimi-k3-improve.md` §4 *Improving per-request speed* |
