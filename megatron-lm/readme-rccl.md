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
- ~30 min of GPU time for the full sweep (2 collectives × 7 N).

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
# focus on the suspect arities
GPU_COUNTS="4 5 6 7 8" bash work/run-rccl-tests.sh

# one collective, one N (smoke test)
COLLECTIVES=all_reduce GPU_COUNTS=5 bash work/run-rccl-tests.sh

# narrower size range (faster, focuses on the message sizes Megatron actually uses)
MIN_BYTES=512M MAX_BYTES=4G bash work/run-rccl-tests.sh
```

### Env knobs

| var | default | purpose |
|-----|---------|---------|
| `GPU_COUNTS` | `2 3 4 5 6 7 8` | which N values to sweep |
| `COLLECTIVES` | `all_reduce all_gather` | which collectives to probe |
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

## Q: What is the GPU-GPU topology on this node? Is it like an NVIDIA DGX? Is there a switch like NVSwitch?

Confirmed from the node itself via `rocm-smi`. Every off-diagonal cell of
`--showtopotype` is `XGMI`, every `--showtopohops` entry is `1`, every
`--showtopoweight` entry is `15`, and every `--showtopoaccess` cell is `True`:

| GPU pair (any i ≠ j) | type | hops | weight | P2P access |
|----------------------|------|------|--------|------------|
| all 28 pairs across {0..7} | XGMI | 1 | 15 | True |

That's `K₈` — fully-connected 8-way mesh. Each of the 8 GPUs has a direct
xGMI link to each of the other 7, point-to-point silicon-to-silicon over
Infinity Fabric. **There is no switch chip** between the GPUs.

This is **not** the NVSwitch model. NVSwitch on DGX H100/H200 is a true
crossbar — NVLink traffic enters a switch chip (4 per DGX H100) that routes
any rank to any other at full bandwidth. The collective library sees an
effectively any-to-any fabric, so the "build a ring on a topology graph"
problem evaporates and non-power-of-2 N looks smooth.

The closer NVIDIA analogue here is **older NVLink-mesh hardware** (DGX-1
P100, NVLink-only A100 single-node sub-meshes): point-to-point links forming
a graph, no central switch, and collective libraries have to do an actual
graph search to construct ring channels. Those systems also showed
non-power-of-2 cliffs broadly similar to what RCCL shows here.

**NUMA note (unrelated to the cliff).** `--showtoponuma` reports GPUs 0–3 on
NUMA node 0 and GPUs 4–7 on NUMA node 1. That is *CPU-side* memory affinity
(2-socket host, 4 GPUs per socket's PCIe root). It affects host↔device
staging and PCIe DMA, but **not** GPU↔GPU xGMI — those go silicon-to-silicon
without traversing the CPU. The uniform `weight=15` / `hops=1` confirms
inter-socket GPU pairs aren't penalized at the fabric layer.

So briefly:

- **Is it 8-way connected like an NVIDIA DGX?** Topologically yes — every GPU
  connects to every other. It's `K₈`.
- **Is there a switch like NVSwitch?** No. Direct point-to-point xGMI silicon
  links, no central switch chip.
- **What it most resembles:** an AMD MI300X UBB (Universal Baseboard), the
  standard 8-GPU module, exposed as a single-node `K₈` mesh.

The second answer is the one that explains the N=5/6/7 cliff observed in
`summary-rccl.md`. The combination of (point-to-point mesh) × (non-power-of-2
ring construction not yet tuned for MI355X) is exactly the condition that
produces the collapse. A future AMD platform with a switched fabric would
lift this constraint at the hardware layer.

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
