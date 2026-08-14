# Base GEMM Benchmark Report

- Date: 2026-08-13 23:21:26
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
| mi355-gpu-33 | 7 | 0 | 0.094571 | 0.14 | 1453.29 | 1064.42 | 1365.33 |
| mi355-gpu-33 | 7 | 1 | 0.097357 | 0.14 | 1411.70 | 1033.96 | 1365.33 |
| mi355-gpu-33 | 7 | 2 | 0.093960 | 0.14 | 1462.74 | 1071.34 | 1365.33 |
| mi355-gpu-33 | 7 | 3 | 0.095632 | 0.14 | 1437.16 | 1052.61 | 1365.33 |
| mi355-gpu-33 | 7 | 4 | 0.095603 | 0.14 | 1437.60 | 1052.93 | 1365.33 |
| mi355-gpu-33 | 7 | 5 | 0.095488 | 0.14 | 1439.33 | 1054.20 | 1365.33 |
| mi355-gpu-33 | 7 | 6 | 0.097850 | 0.14 | 1404.58 | 1028.75 | 1365.33 |
