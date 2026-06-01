# readme-rccl — rccl-tests quick-start

How to download, build, and run the rccl-tests sweep that drives
[`run-rccl-tests.sh`](run-rccl-tests.sh). For *why* this sweep exists and how
to read the results, see [`rccl-tests.md`](rccl-tests.md).

## Prerequisites

- Singularity image at `/home/v89592/shaohao/megatron-lm/megatron-lm.sif`
  (the same one used by `run.sh`). It ships ROCm 6.4.3 + RCCL 2.22 + a
  working `git`, `make`, and `hipcc`, so no host toolchain is needed.
- An MI355X node (or any gfx9 GPU); the script bakes
  `HSA_OVERRIDE_GFX_VERSION=9.4.2` so prebuilt gfx942 kernels run on gfx950
  the same way Megatron does.
- ~30 min of GPU time for the full sweep (5 configs × 2 collectives × 7 N).

## 1. Download + build rccl-tests

One-time. Builds into `work/rccl-tests/build/` — host-bind-mounted, so it
survives container teardown and reboots:

```bash
ROOT=/home/v89592/shaohao/megatron-lm
singularity exec --rocm \
  --bind "$ROOT:$ROOT" \
  "$ROOT/megatron-lm.sif" bash -lc '
    set -euo pipefail
    cd '"$ROOT"'/work
    git clone --depth=1 https://github.com/ROCm/rccl-tests.git rccl-tests
    cd rccl-tests
    make MPI=0 HIP_HOME=/opt/rocm -j
'
```

Build emits gfx906/908/90a/942 + gfx10xx code objects (no gfx950 yet); the
runtime `HSA_OVERRIDE_GFX_VERSION=9.4.2` in the sweep script handles the gap.

Verify:

```bash
ls work/rccl-tests/build/all_reduce_perf work/rccl-tests/build/all_gather_perf
```

## 2. Run the sweep

The script auto-detects `work/rccl-tests/build/` — no env override needed.

```bash
# full sweep, foreground
bash work/run-rccl-tests.sh

# full sweep, background like the Megatron runs
nohup bash work/run-rccl-tests.sh > log.rccl-tests 2>&1 &
```

### Common subsets (fastest debug loop)

```bash
# baseline only — confirm the cliff exists on the same config run.sh uses
CONFIGS=default bash work/run-rccl-tests.sh

# focus on the suspect arities
GPU_COUNTS="4 5 6 7 8" bash work/run-rccl-tests.sh

# one collective, one config, one N (smoke test)
CONFIGS=tree COLLECTIVES=all_reduce GPU_COUNTS=5 bash work/run-rccl-tests.sh

# narrower size range (faster, focuses on the message sizes Megatron actually uses)
MIN_BYTES=512M MAX_BYTES=4G bash work/run-rccl-tests.sh
```

### Env knobs

| var | default | purpose |
|-----|---------|---------|
| `GPU_COUNTS` | `2 3 4 5 6 7 8` | which N values to sweep |
| `COLLECTIVES` | `all_reduce all_gather` | which collectives to probe |
| `CONFIGS` | `default tree ring no_mscll proto_simple` | which RCCL config variants |
| `MIN_BYTES` / `MAX_BYTES` / `STEP_FACTOR` | `16M` / `8G` / `2` | rccl-tests size sweep |
| `ITERS` / `WARMUP` | `20` / `5` | rccl-tests timing knobs |
| `RCCL_TESTS_DIR` | autodetect | force a specific build dir |

## 3. Where the output lands

```
work/logs/rccl_tests_<stamp>/
  rccl_tests_summary.txt        # one-line-per-(coll,config,N): max_size busbw_GB/s
  all_reduce_default_n2.log     # raw rccl-tests stdout per run
  all_reduce_default_n3.log
  ...
  all_gather_proto_simple_n8.log
```

`rccl_tests_summary.txt` is what you compare against the Megatron timer table
in summary-1/2 §2. See [`rccl-tests.md`](rccl-tests.md#how-to-read-the-result)
for the interpretation patterns.

## 4. Troubleshooting

- **"Could not find rccl-tests in the container."** Step 1 didn't run, or
  built to a non-standard place. Fix: rerun step 1, or set
  `RCCL_TESTS_DIR=/path/to/build` and rerun the script.
- **`NCCL WARN NUMA auto balancing enabled ...`** and **`Missing iommu=pt`**.
  Host-kernel settings, can add jitter to the numbers. Root-only to silence:
  `sudo sysctl kernel.numa_balancing=0` and add `iommu=pt` to GRUB cmdline.
  The same warnings appear in the Megatron logs, so the comparison stays fair
  even if you don't change them.
- **One N hangs / crashes.** The script does *not* abort the rest of the
  sweep — it logs `ERR(rc=...)` in the summary row and moves on, same pattern
  as `run.sh`.
- **gfx950-native rebuild.** If a future image has gfx950 code objects, drop
  `HSA_OVERRIDE_GFX_VERSION=9.4.2` from `BASE_CONTAINER_ENV` in the script
  and rebuild rccl-tests inside that image to get native kernels.
