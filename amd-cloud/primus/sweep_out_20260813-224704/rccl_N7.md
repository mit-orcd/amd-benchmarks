# RCCL Benchmark Report

- Date: 2026-08-13 23:26:26
- Cluster: amd-aig-poolside
- World Size: 7
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (7): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33 ...

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N7.md`

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
- output_file: /out/rccl_N7.md
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
- HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6
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
| mi355-gpu-33 | 7 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.041 | 0.544 | 0.040 | 5.237 | 0.04 | r4@mi355-gpu-33 | 0.107 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.044 | 0.052 | 0.039 | 0.067 | 0.08 | r4@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.041 | 0.050 | 0.040 | 0.052 | 0.17 | r6@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.042 | 0.049 | 0.040 | 0.062 | 0.34 | r5@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.050 | 0.057 | 0.049 | 0.071 | 0.56 | r2@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.050 | 0.056 | 0.049 | 0.069 | 1.11 | r6@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.052 | 0.060 | 0.050 | 3.582 | 2.17 | r2@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.063 | 0.069 | 0.062 | 0.072 | 3.57 | r6@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.071 | 0.084 | 0.070 | 5.659 | 6.29 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.081 | 0.085 | 0.080 | 4.392 | 11.03 | r5@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.100 | 0.107 | 0.099 | 6.015 | 17.89 | r6@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.145 | 0.151 | 0.143 | 0.157 | 24.77 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.229 | 0.244 | 0.225 | 3.197 | 31.44 | r1@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.398 | 1.310 | 0.392 | 3.111 | 36.13 | r3@mi355-gpu-33 | 0.017 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.732 | 0.791 | 0.720 | 0.848 | 39.31 | r6@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 1.355 | 1.398 | 1.336 | 1.463 | 42.46 | r3@mi355-gpu-33 | 0.003 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 2.611 | 2.643 | 2.570 | 2.656 | 44.05 | r4@mi355-gpu-33 | 0.004 |
| mi355-gpu-33 | 7 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 5.098 | 5.149 | 5.020 | 5.176 | 45.13 | r5@mi355-gpu-33 | 0.002 |
