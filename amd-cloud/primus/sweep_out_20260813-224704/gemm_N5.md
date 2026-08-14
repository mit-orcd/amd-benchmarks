# Base GEMM Benchmark Report

- Date: 2026-08-13 23:10:49
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
| mi355-gpu-33 | 5 | 0 | 0.094104 | 0.14 | 1460.50 | 1069.71 | 1365.33 |
| mi355-gpu-33 | 5 | 1 | 0.097630 | 0.14 | 1407.75 | 1031.07 | 1365.33 |
| mi355-gpu-33 | 5 | 2 | 0.093671 | 0.14 | 1467.26 | 1074.65 | 1365.33 |
| mi355-gpu-33 | 5 | 3 | 0.096261 | 0.14 | 1427.78 | 1045.73 | 1365.33 |
| mi355-gpu-33 | 5 | 4 | 0.094492 | 0.14 | 1454.51 | 1065.31 | 1365.33 |
