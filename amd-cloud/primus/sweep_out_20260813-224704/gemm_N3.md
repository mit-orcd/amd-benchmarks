# Base GEMM Benchmark Report

- Date: 2026-08-13 23:00:31
- Cluster: amd-aig-poolside
- Benchmark Duration: 10 sec

## GEMM Configuration
- M: 4096
- N: 4096
- K: 4096
- Transpose A: False
- Transpose B: False
- Dtype: bf16

## GEMM Shape
- A: (4096, 4096)
- B: (4096, 4096)
- C: (4096, 4096)

## Metrics
- `avg_time_ms`: average time per matmul (ms)
- `tflops`: total TFLOPS (1e12 ops/sec)
- `bandwidth_gbps`: estimated memory bandwidth usage (GB/s)
- `arith_intensity`: arithmetic intensity (FLOPs per byte)

| host | world | rank | avg_time_ms | tflop | tflops | bandwidth_gbps | arith_intensity |
|---|---|---|---|---|---|---|---|
| mi355-gpu-33 | 3 | 0 | 0.094300 | 0.14 | 1457.46 | 1067.48 | 1365.33 |
| mi355-gpu-33 | 3 | 1 | 0.095584 | 0.14 | 1437.89 | 1053.14 | 1365.33 |
| mi355-gpu-33 | 3 | 2 | 0.094735 | 0.14 | 1450.77 | 1062.58 | 1365.33 |
