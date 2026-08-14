# RCCL Benchmark Report

- Date: 2026-08-13 23:00:10
- Cluster: amd-aig-poolside
- World Size: 2
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (2): mi355-gpu-33, mi355-gpu-33

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N2.md`

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
- output_file: /out/rccl_N2.md
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
- HIP_VISIBLE_DEVICES=0,1
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
| mi355-gpu-33 | 2 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.029 | 0.038 | 0.028 | 0.063 | 0.04 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.029 | 0.039 | 0.028 | 1.790 | 0.07 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.029 | 0.035 | 0.028 | 0.051 | 0.14 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.029 | 0.037 | 0.028 | 0.050 | 0.28 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.030 | 0.040 | 0.029 | 3.297 | 0.55 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.036 | 0.040 | 0.035 | 0.075 | 0.91 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.037 | 0.040 | 0.036 | 0.115 | 1.79 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.037 | 0.042 | 0.036 | 0.052 | 3.55 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.038 | 0.043 | 0.038 | 0.137 | 6.83 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.043 | 0.046 | 0.042 | 0.052 | 12.33 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.050 | 0.056 | 0.050 | 0.074 | 20.77 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.069 | 0.073 | 0.068 | 0.083 | 30.46 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.106 | 0.111 | 0.104 | 0.119 | 39.65 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.180 | 0.186 | 0.178 | 0.190 | 46.63 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.327 | 0.334 | 0.324 | 5.634 | 51.36 | r0@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 0.623 | 0.626 | 0.619 | 0.628 | 53.87 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 1.185 | 1.196 | 1.181 | 1.276 | 56.64 | r0@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 2 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 2.328 | 2.341 | 2.315 | 2.388 | 57.66 | r1@mi355-gpu-33 | 0.000 |
