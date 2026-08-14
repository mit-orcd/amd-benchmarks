# RCCL Benchmark Report

- Date: 2026-08-13 23:21:00
- Cluster: amd-aig-poolside
- World Size: 6
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (6): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N6.md`

- Git Commit: b511d1b6

## Key Arguments
- aggregate_repeat: False
- append: False
- check: False
- cluster: amd-aig-poolside
- command: benchmark
- debug: False
- dtype: bf16
- iters: 100
- max_bytes: 128M
- min_bytes: 1K
- num_sizes: 12
- op: ['all_reduce']
- output_file: /out/rccl_N6.md
- per_iter_trace: False
- per_rank: False
- per_rank_file: 
- repeat: 1
- scale: log2
- sizes: None
- suite: rccl
- trace_file: 
- trace_limit: 0
- trace_ops: 
- trace_sizes: 
- warmup: 20

## Environment
- HIP_VISIBLE_DEVICES=0,1,2,3,4,5
- NCCL_DEBUG=WARN
- RCCL_MSCCL_ENABLE=1
- RCCL_MSCCLPP_ENABLE=0
- RCCL_MSCCLPP_THRESHOLD=1073741824

## Metrics
- `p50_ms` / `p95_ms`: 50/95th percentile of critical-path latency (ms)
- `min_ms` / `max_ms`: min/max latency observed on the critical path (ms)
- `eff_gbps`: per-rank effective bandwidth in GB/s, normalized by collective algorithm factor
- `slowest_rank`: rank with the highest p95 latency (format `rX@host`)
- `rank_p95_spread_ms`: max p95 minus median p95 across ranks, indicating imbalance

| host | world | suite | op | bytes | dtype | repeat | p50_ms | p95_ms | min_ms | max_ms | eff_gbps | slowest_rank | rank_p95_spread_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mi355-gpu-33 | 6 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.039 | 0.047 | 0.038 | 0.060 | 0.04 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.041 | 0.308 | 0.038 | 1.800 | 0.08 | r5@mi355-gpu-33 | 0.006 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.039 | 0.055 | 0.038 | 2.341 | 0.17 | r4@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.039 | 0.048 | 0.038 | 2.353 | 0.35 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.047 | 0.054 | 0.046 | 2.998 | 0.58 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.048 | 0.053 | 0.046 | 0.058 | 1.15 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.047 | 0.054 | 0.046 | 0.085 | 2.30 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.057 | 0.061 | 0.055 | 0.072 | 3.86 | r2@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.063 | 0.073 | 0.062 | 0.128 | 6.89 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.071 | 0.078 | 0.070 | 0.099 | 12.23 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.093 | 0.101 | 0.092 | 6.738 | 18.71 | r4@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.131 | 0.138 | 0.130 | 0.142 | 26.63 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.215 | 3.000 | 0.212 | 4.739 | 32.46 | r1@mi355-gpu-33 | 0.220 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.383 | 0.393 | 0.376 | 2.008 | 36.53 | r4@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.704 | 0.996 | 0.692 | 1.434 | 39.72 | r1@mi355-gpu-33 | 0.020 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 1.304 | 1.360 | 1.290 | 1.574 | 42.88 | r0@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 2.507 | 2.540 | 2.475 | 2.566 | 44.61 | r2@mi355-gpu-33 | 0.004 |
| mi355-gpu-33 | 6 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 4.939 | 4.971 | 4.886 | 5.000 | 45.29 | r1@mi355-gpu-33 | 0.001 |
