# RCCL Benchmark Report

- Date: 2026-08-13 23:05:16
- Cluster: amd-aig-poolside
- World Size: 3
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (3): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N3.md`

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
- output_file: /out/rccl_N3.md
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
- HIP_VISIBLE_DEVICES=0,1,2
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
| mi355-gpu-33 | 3 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.030 | 0.044 | 0.029 | 0.058 | 0.05 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.030 | 0.043 | 0.029 | 0.049 | 0.09 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.030 | 0.042 | 0.029 | 0.054 | 0.18 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.033 | 0.046 | 0.033 | 0.109 | 0.33 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.034 | 0.042 | 0.033 | 0.050 | 0.65 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.034 | 0.043 | 0.033 | 0.529 | 1.27 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.042 | 0.049 | 0.041 | 3.760 | 2.07 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.043 | 0.048 | 0.042 | 0.071 | 4.11 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.043 | 0.050 | 0.042 | 0.114 | 8.14 | r2@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.047 | 0.053 | 0.046 | 0.065 | 15.02 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.056 | 0.061 | 0.055 | 0.069 | 25.16 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.073 | 0.080 | 0.071 | 0.092 | 38.49 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.108 | 0.115 | 0.106 | 6.668 | 51.82 | r0@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.179 | 0.188 | 0.174 | 0.205 | 62.55 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.301 | 0.311 | 0.292 | 4.323 | 74.21 | r0@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 0.558 | 0.566 | 0.549 | 1.411 | 80.23 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 1.073 | 1.164 | 1.059 | 3.506 | 83.41 | r1@mi355-gpu-33 | 0.009 |
| mi355-gpu-33 | 3 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 2.090 | 5.835 | 2.079 | 7.663 | 85.64 | r0@mi355-gpu-33 | 0.001 |
