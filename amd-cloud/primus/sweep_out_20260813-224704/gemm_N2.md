# Base GEMM Benchmark Report

- Date: 2026-08-13 22:55:27
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
| mi355-gpu-33 | 2 | 0 | 0.094001 | 0.14 | 1462.10 | 1070.88 | 1365.33 |
| mi355-gpu-33 | 2 | 1 | 0.096319 | 0.14 | 1426.91 | 1045.10 | 1365.33 |
