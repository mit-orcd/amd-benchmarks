# Base GEMM Benchmark Report

- Date: 2026-08-13 23:26:54
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
| mi355-gpu-33 | 8 | 0 | 0.094378 | 0.14 | 1456.26 | 1066.60 | 1365.33 |
| mi355-gpu-33 | 8 | 1 | 0.097448 | 0.14 | 1410.38 | 1033.00 | 1365.33 |
| mi355-gpu-33 | 8 | 2 | 0.094087 | 0.14 | 1460.76 | 1069.89 | 1365.33 |
| mi355-gpu-33 | 8 | 3 | 0.095635 | 0.14 | 1437.12 | 1052.58 | 1365.33 |
| mi355-gpu-33 | 8 | 4 | 0.094711 | 0.14 | 1451.14 | 1062.85 | 1365.33 |
| mi355-gpu-33 | 8 | 5 | 0.095390 | 0.14 | 1440.80 | 1055.28 | 1365.33 |
| mi355-gpu-33 | 8 | 6 | 0.097075 | 0.14 | 1415.79 | 1036.96 | 1365.33 |
| mi355-gpu-33 | 8 | 7 | 0.093125 | 0.14 | 1475.85 | 1080.94 | 1365.33 |
