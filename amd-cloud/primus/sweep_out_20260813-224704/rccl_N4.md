# RCCL Benchmark Report

- Date: 2026-08-13 23:10:25
- Cluster: amd-aig-poolside
- World Size: 4
- Ops: all_reduce
- Message Sizes: 1.0KB, 2.0KB, 4.0KB, 8.0KB, 16.0KB, 32.0KB, 64.0KB, 128.0KB, 256.0KB, 512.0KB, 1.0MB, 2.0MB, 4.0MB, 8.0MB, 16.0MB, 32.0MB, 64.0MB, 128.0MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (4): mi355-gpu-33, mi355-gpu-33, mi355-gpu-33, mi355-gpu-33

## Command
`primus/cli/main.py benchmark rccl --output-file /out/rccl_N4.md`

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
- output_file: /out/rccl_N4.md
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
- HIP_VISIBLE_DEVICES=0,1,2,3
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
| mi355-gpu-33 | 4 | rccl | all_reduce | 1024 | bfloat16 | 1 | 0.036 | 0.070 | 0.032 | 1.696 | 0.04 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 2048 | bfloat16 | 1 | 0.037 | 0.812 | 0.031 | 2.586 | 0.08 | r1@mi355-gpu-33 | 0.004 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 4096 | bfloat16 | 1 | 0.036 | 0.047 | 0.031 | 1.399 | 0.17 | r2@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 8192 | bfloat16 | 1 | 0.035 | 0.045 | 0.032 | 0.059 | 0.35 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 16384 | bfloat16 | 1 | 0.038 | 0.044 | 0.033 | 0.076 | 0.64 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 32768 | bfloat16 | 1 | 0.037 | 0.045 | 0.033 | 2.201 | 1.34 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 65536 | bfloat16 | 1 | 0.037 | 0.047 | 0.034 | 0.063 | 2.65 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 131072 | bfloat16 | 1 | 0.047 | 0.053 | 0.045 | 0.070 | 4.18 | r0@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 262144 | bfloat16 | 1 | 0.047 | 0.056 | 0.046 | 0.180 | 8.34 | r3@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 524288 | bfloat16 | 1 | 0.048 | 0.055 | 0.047 | 0.064 | 16.27 | r3@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 1048576 | bfloat16 | 1 | 0.055 | 0.062 | 0.053 | 0.069 | 28.84 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 2097152 | bfloat16 | 1 | 0.063 | 0.073 | 0.061 | 0.079 | 49.87 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 4194304 | bfloat16 | 1 | 0.081 | 0.091 | 0.079 | 0.107 | 77.87 | r1@mi355-gpu-33 | 0.000 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 8388608 | bfloat16 | 1 | 0.116 | 0.132 | 0.114 | 4.971 | 108.75 | r1@mi355-gpu-33 | 0.003 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 16777216 | bfloat16 | 1 | 0.188 | 0.205 | 0.186 | 7.491 | 133.63 | r3@mi355-gpu-33 | 0.002 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 33554432 | bfloat16 | 1 | 0.336 | 0.345 | 0.331 | 0.355 | 149.92 | r1@mi355-gpu-33 | 0.001 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 67108864 | bfloat16 | 1 | 0.636 | 1.014 | 0.630 | 1.465 | 158.35 | r2@mi355-gpu-33 | 0.033 |
| mi355-gpu-33 | 4 | rccl | all_reduce | 134217728 | bfloat16 | 1 | 1.208 | 1.277 | 1.198 | 1.331 | 166.62 | r0@mi355-gpu-33 | 0.003 |
