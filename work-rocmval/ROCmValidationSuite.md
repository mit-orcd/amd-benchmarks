# RVS Modules Available on This Install

## Performance / benchmark modules

| Module | What it measures |
|--------|-----------------|
| `gst`   | GEMM FLOPS (GPU stress test, used for TFLOPS sweep) |
| `babel` | Memory bandwidth — BabelStream (STREAM-like copy/scale/add/triad kernels) |
| `iet`   | Peak power — generates EDP stress load and measures watts |
| `perf`  | Similar GEMM stress to `gst` (a second PERF-oriented variant) |

## Qualification / diagnostic modules

| Module | What it checks |
|--------|---------------|
| `mem`  | GPU memory — hardware and soft errors via HIP |
| `peqt` | PCIe bus qualification (link speed, width, capabilities) |
| `pesm` | PCIe state monitor — live PCIe interconnect monitoring |
| `gpup` | Static GPU properties (CU count, VRAM size, clock info, etc.) |
| `rcqt` | ROCm platform config (packages installed, versions, permissions) |
| `smqt` | SBIOS BAR mapping — verifies BAR1/2/4/5 sizes and base addresses |
| `gm`   | GPU monitor — reports state (clocks, temps, utilization) at intervals |
| `tst`  | Thermal stress — monitors throttle temp under load |

## Not available on this build

Missing `.so` files — modules not usable without a rebuild:

- `pbqt` — P2P bandwidth between GPUs
- `pebb` — PCIe end-to-end bandwidth
- `pulse` — (purpose unknown from help output)

## Recommended next runs

- **`babel`** — validate HBM3E bandwidth (8 TB/s claimed per GPU)
- **`iet`** — measure actual power draw under peak compute load
