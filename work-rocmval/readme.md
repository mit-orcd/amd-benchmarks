# RVS TFLOPS Harness — User Guide

A small set of bash scripts wrapped around the locally-built ROCm Validation
Suite (`rvs`) GST module to measure achieved GEMM TFLOPS on AMD GPUs.

| Script | Purpose |
|--------|---------|
| [`run_tflops.sh`](#run_tflopssh) | Sweep `gst` across `{1, 2, 4, 8}` GPUs x all 9 precisions, report per-GPU and aggregate TFLOPS. |
| [`run_tflops_sweep.sh`](#run_tflops_sweepsh--full-18-gpu-sweep) | Thin wrapper that defaults to `1..8` GPUs and tees the run log. |
| [`bf16_tune.sh`](#bf16_tunesh--single-gpu-bf16-shapelayout-sweep) | Single-GPU BF16 variant sweep (matrix shape, layout, output dtype, rotating, batched) for hipBLASLt tuning. |

All three scripts auto-detect a locally-built `rvs` binary (typically in a
sibling `ROCmValidationSuite/install_local/` clone — see the
[Local Build](#local-build-of-rocmvalidationsuite-no-root-no-system-install)
section). No root required.

---

# Installation

## Prerequisites

- An AMD GPU host with ROCm installed at `/opt/rocm` (read-only is fine, no
  root needed for the rest of the install). Verify with:
  ```bash
  /opt/rocm/bin/rocminfo | grep -E "Name|Marketing Name"
  ls /opt/rocm/lib/libhipblaslt*       # required by the gst module
  ```
- `bash`, `git`, `cmake >= 3.16`, `make`, a working C++ compiler. The default
  GCC 8.x toolchain on RHEL 8 is sufficient if you skip the three modules
  noted in [Local Build](#local-build-of-rocmvalidationsuite-no-root-no-system-install)
  (`pebb`, `pbqt`, `pulse`), which need libstdc++ >= 11.
- AMD GPUs visible to `rocm-smi` / `rvs -g`.

## 1. Get the harness scripts

The three scripts (`run_tflops.sh`, `run_tflops_sweep.sh`, `bf16_tune.sh`)
live in a working directory you control — e.g.:

```bash
mkdir -p ~/shaohao/work-rocmval
cd       ~/shaohao/work-rocmval
# Copy run_tflops.sh, run_tflops_sweep.sh, bf16_tune.sh into this directory,
# then make them executable:
chmod +x run_tflops.sh run_tflops_sweep.sh bf16_tune.sh
```

## 2. Download the ROCm Validation Suite source

Clone into a **sibling** directory next to the harness — the scripts'
auto-detection looks for `../ROCmValidationSuite/install_local/bin/rvs`:

```bash
cd ~/shaohao                               # parent of work-rocmval
git clone https://github.com/ROCm/ROCmValidationSuite.git
cd ROCmValidationSuite
```

(If you prefer to keep RVS elsewhere, that's fine — see "RVS binary
auto-detection" below for the alternative locations the scripts probe, or
export `RVS_BIN=/your/path/to/rvs`.)

## 3. Build RVS locally (no root)

Follow the exact commands in [Local Build of
ROCmValidationSuite](#local-build-of-rocmvalidationsuite-no-root-no-system-install).
The end result you want is an executable at:

```
~/shaohao/ROCmValidationSuite/install_local/bin/rvs
```

## 4. Smoke-test the harness

```bash
cd ~/shaohao/work-rocmval
./run_tflops.sh        # auto-detects rvs, runs the full sweep
# or, for a 1-minute sanity check:
GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 ./run_tflops.sh
```

If you see `Using rvs binary: /home/.../ROCmValidationSuite/install_local/bin/rvs`
and a populated `tflops_runs/<timestamp>/summary.txt`, the install is good.

---

# `run_tflops.sh`

Sweeps RVS `gst` (GPU stress test) across `{1, 2, 4, 8}` GPUs and every
supported precision, and reports achieved TFLOPS per GPU plus an aggregate.

## Quick start

```bash
# Full sweep: 9 precisions x 4 GPU counts x ~30s each (~18 min)
./run_tflops.sh
```

Output lands in `tflops_runs/<timestamp>/`:

- `summary.txt` — pretty table
- `summary.csv` — machine-readable, one row per (gpus, precision)
- `<n>x_<prec>.conf` — generated RVS configuration for each run
- `<n>x_<prec>.log` — raw `rvs` stdout for each run

Example summary table:

```
GPUs  Prec   Aggregate    Avg/GPU         Per-GPU peaks (TFLOPS)
--------------------------------------------------------------------------------
1     fp16   1545.61      1545.61         gpu61585=1545.61
8     fp16   12300.00     1537.50         gpu61585=1545.61;gpu29764=1530.20;...
```

## What it does

For each GPU count `N` in `GPU_COUNTS`, the script picks the first `N` GPU IDs
auto-detected from `rvs -g`. For each precision in `PRECISIONS`, it:

1. Emits a generated YAML config (parameters mirror the shipped
   `share/rocm-validation-suite/conf/MI355X/levels/rvs_level_5.conf`:
   matrix sizes, `compute_type`, `out_data_type`, `blas_source: hipblaslt`,
   `target_stress: 0` so the test only measures rather than enforcing a target).
2. Runs `rvs -c <conf>` with `parallel: true`, capturing stdout to a per-run log.
3. Parses `[GPU::  <id>] GFLOPS <n>` lines, takes the **peak** per GPU,
   converts to TFLOPS, and sums to an aggregate.

## Default precisions

| Name | RVS `data_type` | `out_data_type` | `compute_type` | Matrix (a x b x c) |
|------|-----------------|-----------------|----------------|--------------------|
| fp4  | `fp4_r`         | `fp16_r`        | `fp32_r`       | 2048^3, block scaled |
| fp6  | `fp6_e3m2_r`    | `fp16_r`        | `fp32_r`       | 2048^3, block scaled |
| bf6  | `fp6_e2m3_r`    | `fp16_r`        | `fp32_r`       | 2048^3, block scaled |
| fp8  | `fp8_e4m3_r`    | `bf16_r`        | `fp32_r`       | 8192 x 8192 x 16384  |
| bf8  | `fp8_e5m2_r`    | `bf16_r`        | `fp32_r`       | 8192 x 8192 x 16384  |
| fp16 | `fp16_r`        | `fp16_r`        | `fp32_r`       | 8192 x 8192 x 16384  |
| bf16 | `bf16_r`        | `bf16_r`        | `fp32_r`       | 8192 x 8192 x 16384  |
| fp32 | `fp32_r`        | (default)       | `fp32_r`       | 3072^3               |
| fp64 | `fp64_r`        | (default)       | `fp64_r`       | 8192^3               |

## Tunables (environment variables)

| Variable          | Default                              | Meaning |
|-------------------|--------------------------------------|---------|
| `RVS_BIN`         | auto-detected (see below)            | path to `rvs` binary |
| `DURATION_MS`     | `30000`                              | per-test stress duration (ms) |
| `LOG_INTERVAL_MS` | `3000`                               | log interval per test (ms) |
| `GPU_COUNTS`      | `"1 2 4 8"`                          | space-separated list of GPU counts to sweep |
| `PRECISIONS`      | `"fp4 fp6 bf6 fp8 bf8 fp16 bf16 fp32 fp64"` | space-separated subset of precisions |
| `OUT_DIR`         | `<script_dir>/tflops_runs/<timestamp>` | override output directory |

GPU counts greater than the number of detected GPUs are skipped automatically.

The `rvs` binary is auto-detected in this order (first hit wins):

1. `$RVS_BIN` if explicitly set
2. `<script_dir>/install_local/bin/rvs`
3. `<script_dir>/../ROCmValidationSuite/install_local/bin/rvs` (sibling clone)
4. `/opt/rocm/bin/rvs` (system install)
5. `rvs` on `PATH`

## Examples

```bash
# Short sweep (~5 min): all precisions, only 1 and 8 GPU
DURATION_MS=15000 GPU_COUNTS="1 8" ./run_tflops.sh

# Just fp8 across all GPU counts
PRECISIONS=fp8 ./run_tflops.sh

# Single sanity check
GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 ./run_tflops.sh

# Specific output directory
OUT_DIR=/tmp/my_tflops_run ./run_tflops.sh
```

---

# `run_tflops_sweep.sh` — full 1..8 GPU sweep

Thin wrapper around `run_tflops.sh` that defaults `GPU_COUNTS` to every count
from 1 to 8 (instead of the inner script's `1 2 4 8`) and tees the combined
stdout/stderr into `sweep.log` alongside the per-run artifacts. Useful for
producing a full scaling curve in one command.

```bash
# Full sweep: 1..8 GPUs x all 9 precisions (~40 min)
./run_tflops_sweep.sh

# Override the GPU sweep but keep all precisions
GPU_COUNTS="1 4 8" ./run_tflops_sweep.sh

# Restrict precisions
PRECISIONS="fp16 bf16" ./run_tflops_sweep.sh
```

Output lands in `tflops_runs/sweep_<timestamp>/` and contains the same files
as a plain `run_tflops.sh` run (`summary.txt`, `summary.csv`, per-run `.conf`
and `.log`) plus `sweep.log` with the full console output.

Every environment variable accepted by `run_tflops.sh` (`RVS_BIN`,
`DURATION_MS`, `LOG_INTERVAL_MS`, `PRECISIONS`, `OUT_DIR`, ...) is passed
through unchanged.

---

# `bf16_tune.sh` — single-GPU BF16 shape/layout sweep

Companion script that sweeps BF16 GEMM variants on **one GPU** to find which
config knobs squeeze the highest % of peak out of the installed hipBLASLt.
Useful as a tuning step before plugging the winner back into
`run_tflops.sh` for a full 1/2/4/8-GPU run.

Why single-GPU only: at 8 GPUs the 11.2 kW OAM tray throttles for the more
power-dense shapes, masking the kernel-level differences this sweep is
trying to surface.

## Quick start

```bash
# Full sweep: 12 variants x 60s each (~12 min)
./bf16_tune.sh

# Just the layout variants
VARIANT_FILTER='layout' ./bf16_tune.sh

# Faster smoke run on a specific GPU
GPU_ID=4 DURATION_MS=20000 ./bf16_tune.sh
```

Output lands in `bf16_runs/<timestamp>/`:

- `leaderboard.txt` — variants sorted by TFLOPS (descending)
- `leaderboard.csv` — `variant,tflops,pct_of_peak,vs_baseline_pct,note`
- `<variant>.conf` — generated RVS config per variant
- `<variant>.log` — raw `rvs` stdout per variant

## What gets swept

| Variant | What it changes vs baseline |
|---------|-----------------------------|
| `baseline`       | 8192 x 8192 x 16384, NT transpose, BF16 out, rotating=512 (matches `run_tflops.sh`) |
| `fp32_out`       | Removes the BF16 output downcast (`out_data_type: fp32_r`) |
| `NT_layout`      | `transa=0 transb=1` — common DL forward-pass layout |
| `TT_layout`      | `transa=1 transb=1` — alternative path |
| `krich_8k_32k`   | K=32768 — wider K, 2x work per call |
| `krich_4k_64k`   | 4096 x 4096 x 65536 — K-dominant shape |
| `squat_16k_8k`   | 16384 x 16384 x 8192 — same work, M/N-heavy |
| `small_8k_sq`    | 8192^3 — smaller, more cache-friendly |
| `rotating_256`   | `rotating=256` — more cache-resident |
| `rotating_2048`  | `rotating=2048` — force HBM working set |
| `batched_8k_x4`  | `gemm_mode: strided_batched`, batch=4 |
| `batched_4k_x16` | `strided_batched`, 16 small GEMMs/launch |

Define your own by editing the `VARIANTS=( ... )` array near the top of the
script (`name|M|N|K|transa|transb|out_dtype|rotating|gemm_mode|batch|hot_calls|note`).

## Tunables (environment variables)

| Variable           | Default                          | Meaning |
|--------------------|----------------------------------|---------|
| `RVS_BIN`          | auto-detected                    | path to `rvs` binary |
| `DURATION_MS`      | `60000`                          | per-test stress duration (ms) |
| `LOG_INTERVAL_MS`  | `3000`                           | log interval (ms) |
| `RAMP_INTERVAL_MS` | `5000`                           | warmup window excluded from peak (ms) |
| `GPU_ID`           | first detected GPU               | single GPU ID to run on |
| `OUT_DIR`          | `bf16_runs/<timestamp>`          | override output directory |
| `VARIANT_FILTER`   | (unset)                          | regex; only variants whose name matches are run |

## Reading the leaderboard

```
variant                TFLOPS    % peak   vs baseline  note
----------------------------------------------------------------------------------------------
NT_layout             2105.34     84.2%       +3.1%   common DL forward-pass layout
baseline              2042.18     81.7%        0.0%   current run_tflops.sh shape
fp32_out              1987.04     79.5%       -2.7%   removes BF16 downcast on output
...
```

- `% peak` is computed against a hard-coded MI355X BF16 peak of `2500` TFLOPS
  (`BF16_PEAK=2500.0` near the bottom of the script — edit for other parts).
- `vs baseline` is signed percent delta vs the `baseline` variant's peak.
- Per-GPU value is the **peak** GFLOPS seen across log intervals (steady-state
  proxy that ignores ramp-up), converted to TFLOPS.

---

# How aggregation works

- Per-GPU value = **peak** `GFLOPS` value observed across log intervals for that
  GPU ID (a steady-state proxy that ignores ramp-up).
- Aggregate TFLOPS = sum of per-GPU peaks, divided by 1000.
- `gpus_reporting` in `summary.csv` tells you how many GPUs actually emitted
  GFLOPS lines vs. the requested count — if it's lower than requested, check
  the corresponding `<n>x_<prec>.log` for errors.

---

# Requirements

- An executable `rvs` binary at one of the auto-detected locations (see the
  list above) — typically the sibling clone at
  `../ROCmValidationSuite/install_local/bin/rvs`, built per the
  "Local Build" section below.
- ROCm at `/opt/rocm` with `hipblaslt` available (the GST module uses
  `blas_source: hipblaslt` for all precisions).
- AMD GPUs visible to `rvs -g`.

---

# Troubleshooting

- **`aggregate=0.00 TFLOPS  reporting=0/N`**: usually means GPU IDs didn't
  match. Check `rvs -g` output and the generated conf's `device:` line.
  The GPU ID is the **second** number inside `GPU[ N - <id>]`.
- **Action `FAIL`**: open the log for that run; the relevant message from the
  module appears as `RVS-ERROR [GST] ...`.
- **Precisions unsupported by hardware**: e.g. fp4/fp6/bf6 require MI350-class
  silicon; on older parts those runs will error out. Drop them via
  `PRECISIONS="fp8 bf8 fp16 bf16 fp32 fp64"`.

---

# GPU Interconnect (XGMI / Infinity Fabric)

This machine has AMD's equivalent of NVLink. Confirmed via
`rocm-smi --showtopo` and `rocminfo`.

**GPUs:** 8 x AMD Instinct MI355X (gfx950)

**Interconnect: XGMI (Infinity Fabric)**

- All 8 GPUs are connected to each other via **XGMI** (Cross-GPU Memory
  Interconnect), AMD's high-bandwidth direct GPU-to-GPU fabric -- the AMD
  counterpart to NVIDIA NVLink.
- Every pair is **1 hop** apart, i.e. a fully connected fabric (no routing
  through a switch or PCIe).
- Equal weight (15) between all GPU pairs -- no topology asymmetry.

**NUMA layout:**

- GPU 0-3 -> NUMA node 0 (one CPU socket)
- GPU 4-7 -> NUMA node 1 (other CPU socket)
- Cross-socket GPU pairs (e.g. GPU0<->GPU5) are still 1 XGMI hop, but mixing
  host memory across sockets will incur a NUMA penalty on the CPU side.

**In practice:** XGMI on MI355X runs at roughly ~900 GB/s aggregate
bidirectional bandwidth per GPU (comparable to NVLink 4 on H100). This is
what makes collective ops (AllReduce, AllGather) fast for multi-GPU
training on this node.

Quick re-check commands:

```bash
rocm-smi --showtopo        # link type matrix (expect all XGMI)
rocm-smi --showtoponuma    # NUMA affinity per GPU
rocminfo | grep -E "Name|Marketing Name"
```

---

# Local Build of ROCmValidationSuite (No root, no system install)

Use this when ROCm is already present at `/opt/rocm` (read-only is fine) and
you want to build/install RVS entirely inside the project directory. Modules
`pebb`, `pbqt`, and `pulse` include `TransferBench.hpp`, which needs C++20
`<barrier>` (libstdc++ >= 11). On systems with only GCC 8.x they must be
skipped.

Exact commands (run from the RVS source root, e.g.
`/home/v89592/shaohao/ROCmValidationSuite`):

```bash
# 1. Disable the three modules that need libstdc++ >= 11
sed -i 's|^add_subdirectory(pebb.so)|# add_subdirectory(pebb.so) # disabled: needs C++20 <barrier>|' CMakeLists.txt
sed -i 's|^add_subdirectory(pbqt.so)|# add_subdirectory(pbqt.so) # disabled: needs C++20 <barrier>|' CMakeLists.txt
sed -i 's|^add_subdirectory(pulse.so)|# add_subdirectory(pulse.so) # disabled: needs C++20 <barrier>|' CMakeLists.txt

# 2. Configure into a local build dir with a local install prefix
mkdir -p build_local install_local
cmake -S . -B ./build_local \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
  -DCMAKE_INSTALL_PREFIX=$PWD/install_local \
  -DCPACK_PACKAGING_INSTALL_PREFIX=$PWD/install_local

# 3. Build in parallel
make -C ./build_local -j$(nproc)

# 4. Install into the local prefix
make -C ./build_local install

# 5. Smoke-test
./install_local/bin/rvs --help
./install_local/bin/rvs -g
```

Outputs:

- `install_local/bin/rvs` — main binary
- `install_local/lib/rvs/*.so` — modules: `babel, gm, gpup, gst, iet, mem, peqt, perf, pesm, rcqt, smqt, tst`
- `build_local/` — build tree (safe to delete to start fresh)

To re-enable `pebb`, `pbqt`, `pulse`, install a toolchain with
`libstdc++ >= 11` (e.g. `gcc-toolset-11`), revert the three `sed` edits above,
then reconfigure and rebuild.
