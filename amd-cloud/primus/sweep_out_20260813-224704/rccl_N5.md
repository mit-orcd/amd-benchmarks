# RCCL Benchmark Report

- Date: 2026-08-13 23:15:40
- Cluster: amd-aig-poolside
- World Size: 5
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (5): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N5.md`

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
- output_file: /out/rccl_N5.md
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
- HIP_VISIBLE_DEVICES=0,1,2,3,4
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
| mi355-gpu-33 | 5 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.035 | 0.046 | 0.034 | 0.056 | 0.05 | r2@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.035 | 0.046 | 0.034 | 0.057 | 0.09 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.035 | 0.065 | 0.034 | 2.759 | 0.19 | r3@mi355-gpu-33 | 0.009 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.036 | 0.050 | 0.035 | 1.464 | 0.37 | r4@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.042 | 0.051 | 0.041 | 2.934 | 0.62 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.043 | 0.050 | 0.042 | 0.053 | 1.21 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.043 | 0.050 | 0.042 | 0.058 | 2.42 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.052 | 0.058 | 0.051 | 0.063 | 4.05 | r3@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.057 | 0.060 | 0.056 | 0.137 | 7.35 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.069 | 0.579 | 0.068 | 6.883 | 12.15 | r4@mi355-gpu-33 | 0.018 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.088 | 0.093 | 0.087 | 6.240 | 18.97 | r3@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.125 | 0.129 | 0.123 | 0.141 | 26.81 | r4@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.204 | 0.213 | 0.200 | 4.391 | 32.91 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.364 | 1.973 | 0.359 | 2.817 | 36.88 | r4@mi355-gpu-33 | 0.007 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.655 | 0.734 | 0.640 | 1.153 | 41.00 | r3@mi355-gpu-33 | 0.010 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 1.239 | 1.305 | 1.217 | 1.729 | 43.34 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 2.377 | 2.414 | 2.344 | 2.442 | 45.17 | r2@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 5 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 4.692 | 4.736 | 4.631 | 4.747 | 45.77 | r4@mi355-gpu-33 | 0.001 |
