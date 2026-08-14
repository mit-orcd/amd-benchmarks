# ATOM serving benchmark — MI355X

Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on 8 x MI355X (gfx950), ROCm 7.14.
Model: `/model`  ·  Source runs: sweep_20260814_160237, sweep_20260814_161953, sweep_20260814_164903

Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric bandwidth: a load generator drives an OpenAI-compatible server at fixed input/output length while concurrency varies. There is no `dell-cloud/` baseline for this suite -- compare against ATOM's public dashboard, not against this repo.

## Throughput and latency vs concurrency

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.17 | 159.3 | 317.5 | 45.8 | 51.9 | 6.24 | 6.28 | 10 |
| 2 | 0.32 | 301.0 | 596.6 | 36.5 | 97.6 | 6.57 | 6.67 | 20 |
| 4 | 0.63 | 573.6 | 1,152.9 | 42.7 | 89.4 | 6.71 | 6.77 | 40 |
| 8 | 1.23 | 1,146.3 | 2,284.2 | 42.3 | 145.3 | 6.77 | 6.82 | 80 |
| 16 | 2.38 | 2,183.4 | 4,390.1 | 43.1 | 246.3 | 7.11 | 7.22 | 160 |
| 32 | 4.36 | 4,028.9 | 8,045.0 | 43.1 | 475.2 | 7.67 | 7.85 | 320 |
| 64 | 7.87 | 7,258.5 | 14,520.3 | 53.3 | 915.1 | 8.41 | 8.88 | 640 |
| 128 | 12.08 | 11,110.8 | 22,245.9 | 68.2 | 1,782.5 | 11.14 | 11.69 | 1,280 |
| 256 | 16.23 | 14,962.7 | 29,910.9 | 182.2 | 3,566.0 | 16.34 | 17.88 | 2,560 |
| 1 | 0.11 | 101.1 | 201.5 | 71.9 | 129.6 | 9.81 | 9.90 | 10 |
| 2 | 0.20 | 191.1 | 378.7 | 71.2 | 140.9 | 10.36 | 10.43 | 20 |
| 4 | 0.41 | 373.7 | 751.1 | 72.1 | 148.9 | 10.28 | 10.36 | 40 |
| 8 | 0.77 | 711.0 | 1,416.7 | 71.9 | 222.2 | 10.93 | 11.04 | 80 |
| 16 | 1.48 | 1,352.1 | 2,718.5 | 71.8 | 387.6 | 11.49 | 11.72 | 160 |
| 32 | 2.63 | 2,432.4 | 4,857.1 | 72.8 | 753.7 | 12.74 | 13.10 | 320 |
| 64 | 4.47 | 4,124.8 | 8,251.5 | 91.9 | 1,397.7 | 15.00 | 15.60 | 640 |
| 128 | 6.97 | 6,412.5 | 12,839.1 | 110.3 | 2,942.0 | 19.47 | 20.44 | 1,280 |
| 256 | 10.13 | 9,342.4 | 18,675.7 | 154.5 | 5,470.0 | 26.74 | 28.44 | 2,560 |
| 1 | 0.05 | 46.1 | 91.8 | 224.9 | 273.8 | 21.48 | 21.56 | 10 |
| 2 | 0.09 | 87.0 | 172.5 | 251.5 | 446.2 | 22.58 | 22.98 | 20 |
| 4 | 0.17 | 154.2 | 309.9 | 256.5 | 495.1 | 24.98 | 25.32 | 40 |
| 8 | 0.31 | 288.0 | 573.9 | 257.5 | 622.9 | 27.02 | 27.76 | 80 |
| 16 | 0.55 | 500.9 | 1,007.0 | 261.7 | 1,073.4 | 31.21 | 32.24 | 160 |
| 32 | 0.89 | 824.0 | 1,645.4 | 273.9 | 1,896.9 | 37.83 | 40.08 | 320 |
| 64 | 1.37 | 1,258.5 | 2,517.5 | 285.6 | 4,236.4 | 49.91 | 53.10 | 640 |

## Throughput / latency knee

- **Peak output throughput**: 14,962.7 tok/s at concurrency 256 (TTFT med 182.2 ms, TPOT med 16.34 ms).
- No knee detected in the sampled range — throughput was still scaling at the highest concurrency tested. Extend the concurrency list to find the ceiling.
- TPOT grows 8.0x (6.24 -> 49.91 ms) across the sweep while output throughput grows 7.9x — the batching trade-off in one line.

## Metric definitions

| Metric | Meaning |
|---|---|
| TTFT | Time to first token — prefill latency; what a user perceives as 'lag'. |
| TPOT | Time per output token — steady-state decode speed after the first token. |
| ITL | Inter-token latency — per-token gaps, the jitter behind TPOT. |
| output tok/s | Generated tokens per second across all concurrent requests. |
| total tok/s | Input + output tokens per second (prefill work included). |

## Caveats

- **The load generator is co-located with the server**, competing for host CPU. This is the normal ATOM/vLLM benchmarking convention but is not a clean client/server split; absolute req/s at high concurrency is mildly pessimistic.
- `--ignore-eos` forces every request to generate exactly OSL tokens, so throughput is not skewed by early stopping. Good for comparability, not representative of real traffic where output lengths vary.
- `--random-range-ratio` jitters prompt lengths around ISL, so prefix caching cannot trivially inflate results.
- KV cache dtype is fp8 by default (`KV_CACHE_DTYPE`), which affects both memory headroom and achievable concurrency. Record it when comparing runs.

## Source data

| What | Where |
|---|---|
| Per-concurrency JSON | `logs/atom/sweep_*/c<N>.json` |
| Per-concurrency log | `logs/atom/sweep_*/c<N>.log` |
| Sweep summary | `logs/atom/sweep_*/summary.txt` |
| Server log | `logs/atom/server_*/atom_server.log` |
| This table as CSV | `results/atom.csv` |

