# RCCL Benchmark Report

- Date: 2026-08-13 23:32:07
- Cluster: amd-aig-poolside
- World Size: 8
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (8): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33 ...

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N8.md`

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
- output_file: /out/rccl_N8.md
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
- HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
| mi355-gpu-33 | 8 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.031 | 0.625 | 0.029 | 1.703 | 0.06 | r0@mi355-gpu-33 | 0.017 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.029 | 0.056 | 0.028 | 0.087 | 0.12 | r7@mi355-gpu-33 | 0.004 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.030 | 0.053 | 0.029 | 1.676 | 0.24 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.030 | 0.051 | 0.029 | 0.065 | 0.48 | r1@mi355-gpu-33 | 0.004 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.030 | 0.048 | 0.029 | 0.052 | 0.95 | r1@mi355-gpu-33 | 0.003 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.030 | 0.046 | 0.029 | 0.052 | 1.90 | r5@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.031 | 0.046 | 0.030 | 0.052 | 3.72 | r5@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.031 | 0.052 | 0.030 | 2.917 | 7.33 | r7@mi355-gpu-33 | 0.006 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.032 | 0.053 | 0.031 | 4.359 | 14.20 | r2@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.039 | 0.049 | 0.038 | 0.055 | 23.45 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.043 | 0.051 | 0.041 | 5.095 | 42.96 | r5@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.052 | 0.060 | 0.051 | 0.080 | 70.04 | r4@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.062 | 0.067 | 0.060 | 0.088 | 119.20 | r4@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.087 | 0.093 | 0.084 | 0.118 | 168.89 | r2@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.133 | 0.146 | 0.130 | 6.891 | 221.10 | r4@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 0.228 | 2.553 | 0.222 | 5.289 | 258.09 | r3@mi355-gpu-33 | 0.183 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 0.408 | 0.415 | 0.404 | 1.236 | 287.81 | r4@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 8 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 0.659 | 0.771 | 0.654 | 1.134 | 356.34 | r3@mi355-gpu-33 | 0.010 |
