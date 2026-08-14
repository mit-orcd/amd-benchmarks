# Base GEMM Benchmark Report

- Date: 2026-08-13 23:16:05
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
| mi355-gpu-33 | 6 | 0 | 0.094273 | 0.14 | 1457.89 | 1067.79 | 1365.33 |
| mi355-gpu-33 | 6 | 1 | 0.097043 | 0.14 | 1416.26 | 1037.30 | 1365.33 |
| mi355-gpu-33 | 6 | 2 | 0.093757 | 0.14 | 1465.91 | 1073.66 | 1365.33 |
| mi355-gpu-33 | 6 | 3 | 0.096832 | 0.14 | 1419.35 | 1039.56 | 1365.33 |
| mi355-gpu-33 | 6 | 4 | 0.095982 | 0.14 | 1431.93 | 1048.78 | 1365.33 |
| mi355-gpu-33 | 6 | 5 | 0.096315 | 0.14 | 1426.97 | 1045.15 | 1365.33 |
