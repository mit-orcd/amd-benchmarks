# Base GEMM Benchmark Report

- Date: 2026-08-13 23:05:38
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
| mi355-gpu-33 | 4 | 0 | 0.094114 | 0.14 | 1460.35 | 1069.59 | 1365.33 |
| mi355-gpu-33 | 4 | 1 | 0.097270 | 0.14 | 1412.96 | 1034.88 | 1365.33 |
| mi355-gpu-33 | 4 | 2 | 0.094061 | 0.14 | 1461.17 | 1070.19 | 1365.33 |
| mi355-gpu-33 | 4 | 3 | 0.096826 | 0.14 | 1419.44 | 1039.63 | 1365.33 |
