# RCCL Collective Sweep — reproducing summary-rccl.md

Scripts and setup to reproduce the measured results in `summary-rccl.md §1`
on the MI355X 8-GPU node.

| summary-rccl.md content | Script | Output |
|--------------------------|--------|--------|
| §1.1 all-collective × N table | `run-rccl-all.sh` | `logs/rccl_all_<stamp>/` |
| §1.1 sendrecv row (footnote ¹) | `run-rccl-sendrecv.sh` | `logs/rccl_sendrecv_<stamp>/` |
| §1.1 busbw figure | `plot_rccl_busbw.py` | `rccl_busbw_8GiB.png` |

## Layout

```
/home/v89592/shaohao/megatron-lm/
├── megatron-lm.sif              # pre-v26.1 Singularity image (ROCm + RCCL + build tools)
└── work/
    ├── run-rccl-all.sh
    ├── run-rccl-sendrecv.sh
    ├── plot_rccl_busbw.py
    ├── rccl-tests/
    │   └── build/               # compiled rccl-tests binaries (one-time build)
    └── logs/
        ├── rccl_all_<stamp>/    # from run-rccl-all.sh
        └── rccl_sendrecv_<stamp>/  # from run-rccl-sendrecv.sh
```

## Requirements

- `singularity` (or `apptainer`) on the host.
- ROCm host driver; `rocm-smi` must see all 8 GPUs.
- No host C++ toolchain needed — `make` and `hipcc` come from inside the container.
- No sudo needed. Scripts write only under `work/logs/` and `work/rccl-tests/`.

**Note:** These scripts use `megatron-lm.sif` (the pre-v26.1 image), **not**
`megatron-lm-v26.1.sif`. `HSA_OVERRIDE_GFX_VERSION=9.4.2` is set inside both
scripts so gfx942-compiled RCCL kernels run on gfx950 hardware.

## Installation

### 1. Singularity image

If `megatron-lm.sif` does not already exist, build it from the pre-v26.1
ROCm Megatron-LM Docker image (ROCm 6.4.3, RCCL 2.22):

```bash
cd /home/v89592/shaohao/megatron-lm
singularity build megatron-lm.sif docker://rocm/megatron-lm
```

Verify:

```bash
ls -lh /home/v89592/shaohao/megatron-lm/megatron-lm.sif
```

### 2. Clone and build rccl-tests

One-time build inside the container. Binaries land in `work/rccl-tests/build/`,
which is host-bind-mounted and survives container restarts.

```bash
ROOT=/home/v89592/shaohao/megatron-lm
singularity exec --rocm \
  --bind "$ROOT:$ROOT" \
  "$ROOT/megatron-lm.sif" bash -lc "
    set -euo pipefail
    cd $ROOT/work
    git clone --depth=1 https://github.com/ROCm/rccl-tests.git rccl-tests
    cd rccl-tests
    make MPI=0 HIP_HOME=/opt/rocm -j
"
```

Verify the key binaries exist:

```bash
ls /home/v89592/shaohao/megatron-lm/work/rccl-tests/build/*_perf
```

The build emits gfx906/908/90a/942 code objects (no native gfx950 yet);
`HSA_OVERRIDE_GFX_VERSION=9.4.2` in both sweep scripts handles the gap at
runtime.

## Running the sweeps

### All-collective sweep — `run-rccl-all.sh`

Runs all 10 RCCL collectives at N=2..8. Produces the main table in
`summary-rccl.md §1.1`. The sweep takes 45–90 min.

```bash
cd /home/v89592/shaohao/megatron-lm/work
nohup bash run-rccl-all.sh > log.nccl-all 2>&1 &
tail -f log.nccl-all
```

The script auto-detects `work/rccl-tests/build/`. The RCCL env is locked to
`NCCL_ALGO=Ring,Tree`, `NCCL_PROTO=Simple,LL,LL128`, `RCCL_MSCCL_ENABLE=1`
so all per-collective numbers stay comparable to the Megatron baseline.

### Sendrecv-only sweep — `run-rccl-sendrecv.sh`

Fills the sendrecv row (footnote ¹ in §1.1). Run this after
`run-rccl-all.sh` if sendrecv was skipped or killed mid-sweep (as happened
in the original run due to the `alltoallv` N=5 OOM):

```bash
nohup bash run-rccl-sendrecv.sh > log.nccl-sendrecv 2>&1 &
tail -f log.nccl-sendrecv
```

Takes ~5 min (one collective × 7 N values).

### Subset runs (faster debugging)

Both scripts accept env overrides:

```bash
# only the arities that show the cliff
GPU_COUNTS="4 5 6 7 8" bash run-rccl-all.sh

# single collective, single N — smoke test
COLLECTIVES=all_reduce GPU_COUNTS=8 bash run-rccl-all.sh

# narrower size range (faster, still hits the saturation plateau)
MIN_BYTES=512M MAX_BYTES=8G bash run-rccl-all.sh
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_COUNTS` | `2 3 4 5 6 7 8` | which N values to sweep |
| `COLLECTIVES` | all 10 | which collectives to run (skips any missing binary) |
| `MIN_BYTES` / `MAX_BYTES` / `STEP_FACTOR` | `16M` / `8G` / `2` | rccl-tests size sweep |
| `ITERS` / `WARMUP` | `20` / `5` | timing iterations |
| `RCCL_TESTS_DIR` | auto-detect | override binary directory |

## Regenerating the figure

`plot_rccl_busbw.py` has the busbw numbers hardcoded (extracted from
`logs/rccl_all_20260602_121713/` and `logs/rccl_sendrecv_20260602_153246/`).
It requires `matplotlib` on the host Python:

```bash
pip install matplotlib          # if not already installed
python3 /home/v89592/shaohao/megatron-lm/work/plot_rccl_busbw.py
```

Produces `work/rccl_busbw_8GiB.png` (the figure embedded in `summary-rccl.md §1.1`).

To update the figure with new sweep numbers, edit the `data = { ... }` dict
in `plot_rccl_busbw.py` with values from the new `rccl_all_summary.txt` and
`rccl_sendrecv_summary.txt`, then re-run the script.

## Output layout

```
logs/rccl_all_<stamp>/
  rccl_all_summary.txt          # one row per (collective, N): max_size busbw_GB/s
  all_reduce_n2.log             # raw rccl-tests stdout per run
  all_reduce_n3.log
  ...
  sendrecv_n8.log               # if sendrecv was not killed

logs/rccl_sendrecv_<stamp>/
  rccl_sendrecv_summary.txt     # sendrecv rows only
  sendrecv_n2.log .. sendrecv_n8.log
```

The `busbw_GB/s` column in each summary file is the **in-place busbw at the
top-end message size (8 GiB)** — the saturated fabric-ceiling value used
directly in the `summary-rccl.md §1.1` table.
