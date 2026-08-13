# Benchmark Plan — AMD Cloud MI355X node

Reproduce, on the **AMD Cloud** server, the three benchmark suites that
[`../dell-cloud/`](../dell-cloud/) ran on a Dell Cloud server. Everything produced for this
machine — scripts, logs, results — stays under `amd-cloud/`; `dell-cloud/` is read-only
reference material and the source of the comparison baselines.

| # | Suite | Dir here (`amd-cloud/`) | Reference (`dell-cloud/`) | What we run |
|---|-------|-------------------------|---------------------------|-------------|
| A | ROCm Validation Suite (RVS) | `work-rocmval/` | `work-rocmval/` | `gst` TFLOPS sweep + health modules |
| B | RCCL collective sweep | `rccl-tests/` | `rccl-tests/` | `rccl-tests` collectives × N=2..8, algo/proto configs |
| C | Primus | `primus/` | `primus/` | GEMM / attention / RCCL microbenches **+ Megatron-LM llama2-7B pretrain** |

**Explicitly out of scope:** the standalone Megatron-LM training sweep in
`dell-cloud/megatron-lm/` (`run.sh`, `summary*.md`). Part B reproduces
`dell-cloud/rccl-tests/` only — **nothing Megatron-related is built, run, or written in our
`rccl-tests/`.** Megatron-LM is benchmarked **only** through Primus (Part C).

No benchmark in this plan has been executed yet — only read-only probing of the host.

> **Status 2026-08-13: setup complete, awaiting go.** Steps 0 and 1 of §5 are done —
> packages, venv, clones, image pull, both builds, and all 13 scripts are in place and
> syntax-checked. **No benchmark has been run.** See [§8 Setup log](#8-setup-log-2026-08-13)
> for what was actually done and the five places reality differed from this plan.

---

## 0. Host inventory (measured, 2026-08-13)

| Item | Value |
|------|-------|
| GPUs | 8 × AMD Instinct MI355X, `gfx950`, IDs 0–7 |
| CPU | 2 × AMD EPYC 9575F (256 threads) |
| RAM | 3.0 TiB |
| OS | Ubuntu 22.04.5, kernel 6.8.0-65 |
| amdgpu driver | 6.19.14.31400100 |
| ROCm userspace | 7.14 (`/opt/rocm`, HIP 7.14.60850) — **runtime only, dev headers missing** |
| hipBLASLt | present (`/opt/rocm/lib/libhipblaslt.so.1.4`) |
| RCCL | present (`/opt/rocm/lib/librccl.so.1`), `amdrocm-rccl7.14-gfx950` |
| Container runtime | **docker 29.7.2 only** — no singularity, no apptainer, no podman |
| Build tools | gcc/g++ 12.3, cmake 3.31.7, make, OpenMPI (`mpirun`) |
| Host python | 3.10.12 |
| sudo | passwordless ✔ |
| Network | github.com, docker hub, pypi all reachable ✔ |
| Interconnect | XGMI all-to-all, 1 hop between every GPU pair; GPU0-3→NUMA0, GPU4-7→NUMA1 |
| NICs | 8 × RoCE (`rocep*`) present but **irrelevant** — single-node run, IB disabled |

### Filesystems

| Mount | Size | Free | Use |
|-------|------|------|-----|
| `/` (`/dev/sdb2`) | 839 G | **216 G** | repo, scripts, `/var/lib/docker` |
| `/mnt/disk0` | 7.0 T | 5.3 T | **in use by k8s PVCs — do not touch** |
| `/mnt/scratch` | 7.0 T | **6.6 T** | empty, world-writable → container caches + analysis venv |

### Two deviations from the Dell Cloud runs, and why

1. **Docker instead of Singularity/Apptainer.** The `dell-cloud/` scripts are all
   `singularity exec --rocm ... --overlay overlay-megatron.img`. Neither singularity nor
   apptainer nor podman is installed here. Every container invocation is rewritten as
   `docker run`. Bonus: docker containers have a writable root layer, so the 20 GiB
   ext3 overlay hack (`overlay-megatron.img`, needed because `--writable-tmpfs` is only
   16 MiB) is **not needed at all**.
2. **No `HSA_OVERRIDE_GFX_VERSION=9.4.2`.** Dell Cloud set `9.4.2` (gfx942) everywhere
   because its 2025-era images had no gfx950 code objects. This host runs ROCm 7.14 with
   native gfx950 support, and the images we pull are gfx950-native. Setting the override
   would silently run gfx942 kernels on MI355X and *undercount* performance. Leave it unset.
   If a run fails with `No HIP GPUs are available`, that is the signal to reconsider — not before.

---

## 1. What to install

### 1.1 Host packages (`sudo apt`)

> ✅ **Done 2026-08-13 — and it turned out to be a no-op.** All five packages were
> *already installed* (`7.14.0-3`), and `/opt/rocm/include/hip/hip_runtime.h` +
> `/opt/rocm/include/rccl/rccl.h` were both already present. The "runtime-only, dev
> headers missing" finding in §0 was wrong — either the earlier probe checked the wrong
> path or the box was provisioned in between. `hipcc --version` → HIP 7.14.60850,
> AMD clang 23.0.0. No `apt-get` was run.

ROCm 7.14 was believed to be installed **runtime-only** — `/opt/rocm/include/hip/hip_runtime.h`
and `rccl.h` absent, so nothing HIP could compile. Both RVS and rccl-tests need them.
Verified available from `repo.amd.com`:

```bash
sudo apt-get update
sudo apt-get install -y \
  amdrocm-core-dev7.14-gfx950 \    # HIP headers + hipcc dev bits for gfx950
  amdrocm-ccl-dev7.14 \            # rccl.h  (rccl-tests)
  amdrocm-rccl-dev7.14 \           # RCCL dev files
  amdrocm-blas-dev7.14 \           # hipBLASLt headers (RVS gst module)
  amdrocm-hipblas-common-dev7.14
# already installed, listed for completeness:
#   libpci-dev, libyaml-cpp-dev, doxygen, cmake, g++-12
```

Verify before moving on:

```bash
ls /opt/rocm/include/hip/hip_runtime.h /opt/rocm/include/rccl/rccl.h
hipcc --version
```

### 1.2 Python (host, for analysis only)

```bash
python3 -m venv /mnt/scratch/shaohao/venv
/mnt/scratch/shaohao/venv/bin/pip install matplotlib pandas tabulate
```

`dell-cloud/primus/README.md` claims `generate_report.py` needs Python ≥ 3.11. **It does not** —
I parsed it under 3.10.12 and it compiles clean (only `from __future__ import annotations`
plus PEP-585 generics). Host `python3` is fine; no deadsnakes PPA needed.

### 1.3 Container images (`docker pull`)

| Image | Compressed | Used by | Why this tag |
|-------|-----------|---------|--------------|
| `rocm/primus:v26.5` | 14.1 GB | Part C | Newest (2026-07-23), ROCm 7.14 + PyTorch 2.12 + TE 2.15 — **matches the host driver exactly**, native gfx950 |
| `rocm/primus:v25.9_gfx950` | 25.1 GB | Part C fallback | The exact image the Dell Cloud run used; primus-turbo v0.1.0 prebuilt. Pull **only if** v26.5 fails |

The Dell Cloud run used `v26.3` (for the microbenches) + `v25.9_gfx950` (for Megatron), because v26.3
lacked primus-turbo for gfx950. v26.5 is three releases newer and built on the same ROCm as
this host, so try **one image for everything** first — that is 14 GB instead of 40 GB, and
removes the two-image split that made the Dell Cloud Megatron runs a separate rerun script.

Parts A and B need **no image at all** — they build native gfx950 binaries on the host.

> ✅ **Done 2026-08-13.** `rocm/primus:v26.5` pulled — **54.8 GB on disk**, not the
> 35–45 GB estimated. `/` went 214 G → 161 G free. The go/no-go gate passed on the first
> try, so **`v25.9_gfx950` was never pulled and is not needed**:
>
> ```
> torch 2.12.0+rocm7.15.0a20260720
> arch_list contains gfx950 ✔   (also gfx942, gfx90a, gfx1100+…)
> torch.cuda.device_count() = 8 ✔
> ```
>
> Note the image ships ROCm **7.15**, one minor ahead of the host's 7.14 — forward-compatible
> with the 6.19 driver, and it is the reason the gfx950 code objects are native.

> ⚠ **Disk is the tightest constraint.** `/` has 216 GB free and `/var/lib/docker` lives
> there. `docker system df` reports an existing `rocm/atom-dev` image (106 GB) plus a
> **stopped container `912e62742a60` holding 283 GB of reclaimable layers — that is not
> ours. Do not `docker system prune`.** v26.5 unpacks to roughly 35–45 GB, which fits, but
> pull one image at a time and re-check `df -h /` between pulls. If space runs out, the
> options in order of preference are (a) ask the owner about the 283 GB container,
> (b) move docker's `data-root` to `/mnt/scratch/docker` via `/etc/docker/daemon.json` +
> daemon restart — **this needs the user's OK first**, since restarting dockerd disturbs
> the other container.

---

## 2. What to download

| What | From | To | Size |
|------|------|-----|------|
| ROCmValidationSuite source | `https://github.com/ROCm/ROCmValidationSuite.git` | `amd-cloud/work-rocmval/ROCmValidationSuite` | ~50 MB |
| rccl-tests source | `https://github.com/ROCm/rccl-tests.git` | `amd-cloud/rccl-tests/src` | ~5 MB |
| Primus source | `https://github.com/AMD-AIG-AIMA/Primus.git` | `amd-cloud/primus/Primus` | ~100 MB |
| Container images | docker hub (§1.3) | `/var/lib/docker` | 14–40 GB |

Reference scripts and baselines need no download — they are already in this repo at
`../dell-cloud/`. Upstream source trees and their build outputs are cloned **inside**
`amd-cloud/` (so everything for this server lives here) but are git-ignored via
`amd-cloud/.gitignore`; only our own scripts, logs, and results are tracked.

**No datasets and no model weights are downloaded.** Megatron runs with `mock_data: true`
and the tokenizer is patched `Llama2Tokenizer → NullTokenizer` (+ `vocab_size: 32000`), so
`HF_HUB_OFFLINE=1` works and nothing is fetched from HuggingFace. This is the same trick
`dell-cloud/primus/rerun_megatron_gfx950_v2.sh` uses, and it is the difference between a working
offline run and a tokenizer download failure.

---

## 3. Directory layout

Scripts, logs, and results are tracked in this repo — matching how `dell-cloud/` keeps its
own logs. Only bulk caches (docker layers, Triton/HF/pip JIT caches) go to `/mnt/scratch`,
since they are regenerable and would otherwise dwarf the repo.

```
/home/amd/shaohao/amd-benchmarks/
├── README.md                    # server index
├── dell-cloud/                  # reference: prior runs on the Dell Cloud server
└── amd-cloud/                   # ← THIS SERVER; $BENCH_ROOT
    ├── plan.md                  # this file
    ├── .gitignore               # ignores upstream clones + build trees
    ├── common/
    │   └── env.sh               # shared paths, GPU list, RCCL env, docker helper
    ├── work-rocmval/            # PART A
    │   ├── ROCmValidationSuite/ # cloned upstream + build_local/ + install_local/  [ignored]
    │   ├── run_tflops.sh        # copied from ../../dell-cloud/, patched
    │   ├── run_tflops_sweep.sh  # NEW
    │   ├── run_rvs_health.sh    # NEW
    │   └── analyze_rvs.py       # NEW
    ├── rccl-tests/              # PART B — RCCL collectives ONLY, no Megatron
    │   ├── src/                 # cloned ROCm/rccl-tests -> src/build/  [ignored]
    │   ├── run-rccl-all.sh      # NEW
    │   ├── run-rccl-sendrecv.sh # NEW
    │   ├── run-rccl-configs.sh  # NEW
    │   ├── analyze_rccl.py      # NEW
    │   └── plot_rccl_busbw.py   # NEW
    ├── primus/                  # PART C
    │   ├── Primus/              # cloned upstream  [ignored]
    │   ├── run_gpu_scan.sh      # NEW (docker port of the dell-cloud script)
    │   ├── run_full_sweep.sh    # NEW (docker port)
    │   ├── run_megatron.sh      # NEW (docker port, GBS bug fixed up front)
    │   └── generate_report.py   # copied from ../../dell-cloud/primus/, unmodified
    ├── logs/{rvs,rccl,primus}/  # per-run driver logs + summaries (tracked)
    └── results/                 # final markdown/CSV/PNG deliverables (tracked)

/mnt/scratch/shaohao/            # regenerable bulk only — never the source of truth
├── cache/{triton,hf,torch,pip}/ # bind-mounted into containers
└── venv/                        # analysis python
```

`amd-cloud/.gitignore`:

```gitignore
# upstream source trees + build outputs — cloned here, not tracked here
work-rocmval/ROCmValidationSuite/
rccl-tests/src/
primus/Primus/
**/__pycache__/
```

---

## 4. `common/env.sh` — shared setup

```bash
#!/usr/bin/env bash
# Shared config for all three benchmark parts. Source, don't execute.
export BENCH_ROOT=/home/amd/shaohao/amd-benchmarks/amd-cloud
export REF_ROOT=/home/amd/shaohao/amd-benchmarks/dell-cloud   # prior Dell Cloud results
export SCRATCH=/mnt/scratch/shaohao
export LOG_ROOT=$BENCH_ROOT/logs          # tracked in-repo, like dell-cloud/*/logs
export CACHE_ROOT=$SCRATCH/cache          # regenerable bulk, off-repo
export PY=$SCRATCH/venv/bin/python
export ROCM_PATH=/opt/rocm
export NGPU=8
export RVS_BIN=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/bin/rvs
mkdir -p "$LOG_ROOT"/{rvs,rccl,primus} "$CACHE_ROOT"/{triton,hf,torch,pip}

# RCCL env — single node, XGMI only, IB off. Mirrors dell-cloud's CONTAINER_ENV
# block so our numbers stay comparable to the published summaries.
# NOTE: HSA_OVERRIDE_GFX_VERSION is deliberately NOT set (native gfx950).
rccl_env() {
  cat <<'EOF'
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
RCCL_MSCCL_ENABLE=1
NCCL_PROTO=Simple,LL,LL128
NCCL_ALGO=Ring,Tree
NCCL_DEBUG=WARN
EOF
}

# docker run wrapper: GPU passthrough + caches + host networking.
# Usage: dgpu <image> <devs csv> <extra -e KEY=VAL ...> -- <command...>
dgpu_args() {
  echo --rm --network host --ipc host --shm-size 64g \
       --device /dev/kfd --device /dev/dri \
       --group-add video --group-add render \
       --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 --ulimit stack=67108864 \
       -v "$CACHE_ROOT/triton:/root/.triton" \
       -v "$CACHE_ROOT/hf:/root/.cache/huggingface" \
       -v "$CACHE_ROOT/torch:/root/.cache/torch" \
       -v "$CACHE_ROOT/pip:/root/.cache/pip"
}

# Fail fast if someone else is using the GPUs.
assert_gpus_idle() {
  local busy
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$' || true)
  [[ "${busy:-0}" -eq 0 ]] || { echo "WARNING: $busy GPU(s) busy — another workload is running"; }
}
```

---

## PART A — ROCm Validation Suite

Reproduces `work-rocmval/summary.md` and `summary-sweep.md`.

### A.1 Build

RVS is not shipped in this ROCm install (`/opt/rocm/bin/rvs` absent), so build from source
into a local prefix, no root:

> ✅ **Done 2026-08-13 — built clean, all 15 modules.** One extra step the plan missed:
> RVS 1.7.8's `CMakeLists.txt:644` hard-fails with *"TransferBench submodule not
> initialised"*, so `git submodule update --init --recursive` is **required** before cmake
> (pulls TransferBench `5fbfa95`, ~rocm-7.2.4). Added to the block below.
>
> The GCC-12 bet paid off: `pebb`, `pbqt` and `pulse` all compiled, so we get the PCIe/P2P
> bandwidth tests the Dell Cloud run had to `sed` out. No `sed` workaround was needed.
> Modules built: `babel gm gpup gst iet mem pbqt pebb peqt perf pesm pulse rcqt smqt tst`.
> `rvs -g` lists all 8 MI355X (device 30115). Build log: `scratchpad/rvs_build.log`.

```bash
cd $BENCH_ROOT/work-rocmval
git clone https://github.com/ROCm/ROCmValidationSuite.git
cd ROCmValidationSuite
git submodule update --init --recursive   # REQUIRED: TransferBench, else cmake fails
mkdir -p build_local install_local
cmake -S . -B build_local \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
  -DCMAKE_INSTALL_PREFIX=$PWD/install_local \
  -DCPACK_PACKAGING_INSTALL_PREFIX=$PWD/install_local
make -C build_local -j"$(nproc)"
make -C build_local install
./install_local/bin/rvs -g       # must list 8 GPUs
```

> `dell-cloud/work-rocmval/readme.md` sed-disables the `pebb`, `pbqt`, and `pulse` modules
> because `TransferBench.hpp` needs C++20 `<barrier>` and that RHEL 8 host only had GCC 8.
> **We have GCC 12 — build all modules**, which gains us the PCIe/P2P bandwidth tests
> (`pebb`, `pbqt`) the Dell Cloud run had to skip. If cmake errors on those three anyway,
> apply the same three `sed` lines to `CMakeLists.txt` and rebuild.

### A.2 `work-rocmval/run_tflops.sh` — reuse from `dell-cloud/`

Copy `../dell-cloud/work-rocmval/run_tflops.sh` verbatim. It emits a `gst` YAML per
(GPU-count, precision), runs `rvs -c`, parses `[GPU:: id] GFLOPS n` and takes the per-GPU
peak. Sweeps 9 precisions (`fp4 fp6 bf6 fp8 bf8 fp16 bf16 fp32 fp64` — all supported on
gfx950) × `GPU_COUNTS`. Two changes needed:

Its `rvs` auto-detection probes `<script_dir>/install_local/bin/rvs` and
`<script_dir>/../ROCmValidationSuite/install_local/bin/rvs`; in our layout RVS is a *child*
of the script dir, so neither hits. `common/env.sh` therefore exports the path explicitly:

```bash
export RVS_BIN=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/bin/rvs
```

And redirect its output under the tracked log tree:

```bash
sed -i 's#^OUT_DIR="${OUT_DIR:-.*}"#OUT_DIR="${OUT_DIR:-$LOG_ROOT/rvs/tflops_$TS}"#' run_tflops.sh
```

Smoke test first: `GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 ./run_tflops.sh` (~1 min).

### A.3 `work-rocmval/run_tflops_sweep.sh` — NEW

The repo documents this script but never committed it. Thin wrapper for the full 1..8 curve:

```bash
#!/usr/bin/env bash
# Full 1..8 GPU × 9 precision RVS gst sweep (~40 min at DURATION_MS=30000).
set -euo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$LOG_ROOT/rvs/sweep_$TS"; mkdir -p "$OUT"
GPU_COUNTS="${GPU_COUNTS:-1 2 3 4 5 6 7 8}" \
DURATION_MS="${DURATION_MS:-30000}" \
OUT_DIR="$OUT" ./run_tflops.sh 2>&1 | tee "$OUT/sweep.log"
echo "results: $OUT"
```

### A.4 `work-rocmval/run_rvs_health.sh` — NEW

The TFLOPS sweep only exercises `gst`. This is the "is the box healthy" gate that
`dell-cloud/rccl-tests/rccl-tests.md §Q` argues for — it validates the floor *beneath* RCCL, so a
clean result here means a later RCCL cliff is an algorithm problem, not hardware.

> ⚠️ **Two corrections applied to the script as written below** (the committed
> `run_rvs_health.sh` has them; this listing is kept for the record):
>
> 1. **There is no `pqt` module in RVS 1.7.8.** The plan's `"pqt:pqt_single.conf:peer-to-peer
>    XGMI bandwidth"` entry names a module and a conf that do not exist — it would have
>    silently `SKIP`ped, and the `find` fallback can't rescue it either (`pqt*` doesn't match
>    `pbqt`). Peer-to-peer / XGMI bandwidth is **`pbqt`** (P2P Benchmark and Qualification
>    Tool); `pebb` is host↔device. The plan also mislabelled `pbqt` as "PCIe bidirectional".
>    This matters: `pbqt` is the module Part B's cliff attribution depends on, so as written
>    the plan would have skipped the one test that licenses the "algorithm, not fabric"
>    conclusion. The `pqt` entry is dropped and the two descriptions corrected.
> 2. **Confs resolve MI355X-first.** `conf/MI355X/` ships tuned `babel.conf`,
>    `pebb_single.conf`, `pbqt_single.conf`, `gst_single.conf`, `iet_stress.conf` — prefer
>    those over the generic ones, falling back to `conf/<name>` then a glob.
>
> Confirmed present: `conf/MI355X/levels/rvs_level_{1..5}.conf`, so the level-config loop at
> the end of the script will fire.

```bash
#!/usr/bin/env bash
# RVS health modules: memory, HBM bandwidth, P2P/XGMI, PCIe, power, config checks.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
RVS=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/bin/rvs
CONF=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/share/rocm-validation-suite/conf
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rvs/health_$TS; mkdir -p "$OUT"
SUM=$OUT/health_summary.txt

# module : shipped conf : what it proves
MODULES=(
  "gpup:gpup_single.conf:GPU properties / config registers"
  "peqt:peqt_single.conf:PCIe qualification"
  "smqt:smqt_single.conf:SBIOS/VRAM mapping"
  "rcqt:rcqt_single.conf:ROCm package + user/group checks"
  "mem:mem.conf:HBM error/pattern test"
  "babel:babel.conf:HBM bandwidth"
  "pqt:pqt_single.conf:peer-to-peer XGMI bandwidth"
  "pebb:pebb_single.conf:PCIe host<->device bandwidth"
  "pbqt:pbqt_single.conf:PCIe bidirectional / P2P"
  "iet:iet_single.conf:sustained power / EDP"
)
{ echo "RVS health run $TS"; echo "rvs: $RVS"; echo; } | tee "$SUM"
for entry in "${MODULES[@]}"; do
  IFS=: read -r mod conf desc <<<"$entry"
  c="$CONF/$conf"; log="$OUT/${mod}.log"
  [[ -f "$c" ]] || { c=$(find "$CONF" -name "${mod}*.conf" | head -1); }
  [[ -f "$c" ]] || { echo "SKIP $mod (no conf found)" | tee -a "$SUM"; continue; }
  echo "----- $mod ($desc) conf=$(basename "$c") -----" | tee -a "$SUM"
  timeout 900 "$RVS" -c "$c" -d 3 >"$log" 2>&1
  rc=$?
  pass=$(grep -c "pass.*true\|RESULT.*pass" "$log" 2>/dev/null || echo 0)
  fail=$(grep -ci "RVS-ERROR\|FAIL" "$log" 2>/dev/null || echo 0)
  echo "  rc=$rc pass_lines=$pass error_lines=$fail log=$log" | tee -a "$SUM"
done
# MI355X shipped level configs, if present
for lvl in "$CONF"/MI355X/levels/rvs_level_*.conf; do
  [[ -f "$lvl" ]] || continue
  n=$(basename "$lvl" .conf); echo "----- $n -----" | tee -a "$SUM"
  timeout 1800 "$RVS" -c "$lvl" >"$OUT/$n.log" 2>&1
  echo "  rc=$? log=$OUT/$n.log" | tee -a "$SUM"
done
echo "results: $OUT"
```

### A.5 `work-rocmval/analyze_rvs.py` — NEW

```python
#!/usr/bin/env python3
"""Aggregate RVS gst sweeps into results/rvs_tflops.{md,csv} + a scaling table.

Usage: analyze_rvs.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>
Reads each run dir's summary.csv (written by run_tflops.sh); if absent, falls
back to re-parsing the raw <n>x_<prec>.log files for '[GPU:: <id>] GFLOPS <v>'.
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

# MI355X vendor peaks (TFLOPS, dense, no sparsity) — used for % of peak.
PEAK = {"fp4": 10000.0, "fp6": 5000.0, "bf6": 5000.0, "fp8": 5000.0,
        "bf8": 5000.0, "fp16": 2500.0, "bf16": 2500.0, "fp32": 157.3, "fp64": 78.6}
GFLOPS_RE = re.compile(r"\[GPU::\s*(\d+)\]\s+.*?GFLOPS\s+([\d.]+)")

def from_logs(d: Path):
    rows = []
    for log in sorted(d.glob("*x_*.log")):
        m = re.match(r"(\d+)x_(\w+)\.log", log.name)
        if not m: continue
        n, prec = int(m.group(1)), m.group(2)
        peaks = defaultdict(float)
        for gid, val in GFLOPS_RE.findall(log.read_text(errors="replace")):
            peaks[gid] = max(peaks[gid], float(val))
        if not peaks: continue
        agg = sum(peaks.values()) / 1000.0
        rows.append({"gpus": n, "precision": prec, "aggregate_tflops": agg,
                     "avg_per_gpu_tflops": agg / len(peaks),
                     "gpus_reporting": len(peaks)})
    return rows

def load(d: Path):
    csvf = d / "summary.csv"
    if csvf.exists():
        with csvf.open() as f:
            return [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    return from_logs(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in a.dirs:
        rows += [{**r, "run": d.name} for r in load(d)]
    if not rows:
        sys.exit("no RVS results found")
    for r in rows:
        r["gpus"] = int(r["gpus"]); r["aggregate_tflops"] = float(r["aggregate_tflops"])
        r["avg_per_gpu_tflops"] = float(r.get("avg_per_gpu_tflops") or
                                        r["aggregate_tflops"] / r["gpus"])
        p = PEAK.get(r["precision"])
        r["pct_of_peak"] = round(100 * r["avg_per_gpu_tflops"] / p, 1) if p else ""
    with (a.out / "rvs_tflops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # markdown: precision × GPU-count matrix of aggregate TFLOPS + scaling efficiency
    precs = sorted({r["precision"] for r in rows}, key=lambda p: list(PEAK).index(p) if p in PEAK else 99)
    ns = sorted({r["gpus"] for r in rows})
    idx = {(r["precision"], r["gpus"]): r for r in rows}
    L = ["# RVS `gst` TFLOPS — MI355X ×8 (gfx950, ROCm 7.14)", "",
         "## Aggregate TFLOPS", "", "| Precision | " + " | ".join(f"N={n}" for n in ns) +
         " | % peak @N=1 | scaling N=8/N=1 |", "|" + "---|" * (len(ns) + 3)]
    for p in precs:
        cells = [f'{idx[(p, n)]["aggregate_tflops"]:.1f}' if (p, n) in idx else "-" for n in ns]
        one, eight = idx.get((p, 1)), idx.get((p, 8))
        pk = f'{one["pct_of_peak"]}%' if one and one["pct_of_peak"] != "" else "-"
        sc = f'{eight["aggregate_tflops"]/one["aggregate_tflops"]/8*100:.0f}%' if one and eight else "-"
        L.append(f"| {p} | " + " | ".join(cells) + f" | {pk} | {sc} |")
    L += ["", "## Per-GPU average TFLOPS", "",
          "| Precision | " + " | ".join(f"N={n}" for n in ns) + " |", "|" + "---|" * (len(ns) + 1)]
    for p in precs:
        L.append(f"| {p} | " + " | ".join(
            f'{idx[(p, n)]["avg_per_gpu_tflops"]:.1f}' if (p, n) in idx else "-" for n in ns) + " |")
    L += ["", "Per-GPU value = peak GFLOPS across log intervals (ignores ramp-up); "
              "aggregate = sum of per-GPU peaks. A per-GPU drop as N grows on a "
              "power-dense precision is OAM-tray power capping, not a kernel regression."]
    (a.out / "rvs_tflops.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rvs_tflops.md and .csv ({len(rows)} rows)")

if __name__ == "__main__":
    main()
```

### A.6 Part A run order

```bash
source common/env.sh
cd work-rocmval
GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 ./run_tflops.sh   # ~1 min smoke
nohup ./run_tflops_sweep.sh   > $LOG_ROOT/rvs/sweep.out  2>&1 &  # ~40 min
nohup ./run_rvs_health.sh     > $LOG_ROOT/rvs/health.out 2>&1 &  # ~20 min (run AFTER, not concurrently)
$PY analyze_rvs.py $LOG_ROOT/rvs/sweep_* -o $BENCH_ROOT/results
```

**Never run two GPU benchmarks concurrently** — they contend for the same 8 GPUs and the
same power envelope, and every number becomes meaningless.

---

## PART B — RCCL collective sweep → local dir `rccl-tests/`

Reproduces [`../dell-cloud/rccl-tests/`](../dell-cloud/rccl-tests/) — its `summary-rccl.md §1`
sweep and the `rccl-tests.md` config sweep — **and only that**. No Megatron image, source,
config, or run script belongs in this directory. `dell-cloud/megatron-lm/run.sh` is not
reproduced anywhere; Megatron-LM is Part C's job, via Primus.

`dell-cloud/rccl-tests/` has the docs and the logs but **not** the scripts (`run-rccl-all.sh`,
`run-rccl-sendrecv.sh`, `run-rccl-tests.sh`, `plot_rccl_busbw.py` are all referenced but
uncommitted) — so all of Part B is written from scratch against the documented interface.

The headline question this part answers: **is there a non-power-of-2 collective cliff at
N=5/6/7 on this box, and does any RCCL knob recover it?**

### B.1 Build rccl-tests (host-native, gfx950)

```bash
mkdir -p $BENCH_ROOT/rccl-tests && cd $_
git clone --depth=1 https://github.com/ROCm/rccl-tests.git src
cd src && make MPI=0 HIP_HOME=/opt/rocm -j"$(nproc)"
ls build/*_perf   # all_reduce_perf, all_gather_perf, alltoall_perf, ... 10 binaries
```

> ✅ **Done 2026-08-13.** Built clean against host ROCm 7.14 → **12** `*_perf` binaries
> (plan said 10; the extras are `all_reduce_bias_perf` and `hypercube_perf`). All 10
> collectives named in `run-rccl-all.sh` have a binary. Build log: `scratchpad/rccl_build.log`.

`MPI=0` is correct — single node, and rccl-tests spawns N threads/GPUs itself via `-g N`.
Building on the host against ROCm 7.14 gives **native gfx950 code objects**, which is
strictly better than the Dell Cloud container build (gfx906/908/90a/942 only, papered over with
`HSA_OVERRIDE_GFX_VERSION=9.4.2`). Absolute bandwidth numbers may therefore come out
*higher* than the published `summary-rccl.md` — note that when comparing.

### B.2 `rccl-tests/run-rccl-all.sh` — NEW

All 10 collectives × N=2..8, 16 MiB → 8 GiB, dell-cloud's locked RCCL env.

```bash
#!/usr/bin/env bash
# All-collective RCCL sweep, N=2..8. Reproduces summary-rccl.md §1.1.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
BIN="${RCCL_TESTS_DIR:-$BENCH_ROOT/rccl-tests/src/build}"
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rccl/rccl_all_$TS; mkdir -p "$OUT"
SUM=$OUT/rccl_all_summary.txt

GPU_COUNTS="${GPU_COUNTS:-2 3 4 5 6 7 8}"
COLLECTIVES="${COLLECTIVES:-all_reduce all_gather reduce_scatter broadcast reduce gather scatter alltoall alltoallv sendrecv}"
MIN_BYTES="${MIN_BYTES:-16M}"; MAX_BYTES="${MAX_BYTES:-8G}"; STEP_FACTOR="${STEP_FACTOR:-2}"
ITERS="${ITERS:-20}"; WARMUP="${WARMUP:-5}"

while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done < <(rccl_env)
assert_gpus_idle
{ echo "RCCL all-collective sweep $TS"; echo "bins: $BIN"; echo "env: $(rccl_env | tr '\n' ' ')";
  echo; printf '%-14s %3s %10s %14s\n' collective N max_size busbw_GB/s; } | tee "$SUM"

for coll in $COLLECTIVES; do
  exe="$BIN/${coll}_perf"
  [[ -x "$exe" ]] || { echo "SKIP $coll (no binary)" | tee -a "$SUM"; continue; }
  # alltoallv at N=5 OOMed in the reference run (each rank allocates N× buffers);
  # cap it so the sweep completes instead of dying mid-way.
  maxb="$MAX_BYTES"; [[ "$coll" == alltoallv || "$coll" == alltoall ]] && maxb="${ALLTOALL_MAX:-4G}"
  for N in $GPU_COUNTS; do
    log="$OUT/${coll}_n${N}.log"
    timeout 1200 "$exe" -b "$MIN_BYTES" -e "$maxb" -f "$STEP_FACTOR" \
        -g "$N" -n "$ITERS" -w "$WARMUP" -c 1 >"$log" 2>&1
    rc=$?
    # last data row: cols are size count type redop root time algbw busbw #wrong (in-place = cols 10..12)
    read -r size busbw < <(awk '/^ *[0-9]/ {sz=$1; bw=$(NF-1)} END {print sz, bw}' "$log")
    printf '%-14s %3s %10s %14s   rc=%s\n' "$coll" "$N" "${size:--}" "${busbw:--}" "$rc" | tee -a "$SUM"
  done
done
echo "results: $OUT"
```

### B.3 `rccl-tests/run-rccl-sendrecv.sh` — NEW

The repo needed this because `sendrecv` got killed when `alltoallv` OOMed at N=5. Our
`ALLTOALL_MAX` cap should prevent that, but keep the script for reruns:

```bash
#!/usr/bin/env bash
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
COLLECTIVES=sendrecv \
OUT_TAG=sendrecv \
exec "$BENCH_ROOT/rccl-tests/run-rccl-all.sh"
```

### B.4 `rccl-tests/run-rccl-configs.sh` — NEW

The diagnostic sweep from `rccl-tests.md`: 5 RCCL configs × 2 collectives × N=2..8 (70 runs,
~30–45 min). This is what tells you *which knob* recovers a cliff.

```bash
#!/usr/bin/env bash
# 5 RCCL configs × {all_reduce, all_gather} × N=2..8 — isolates the N=5/6/7 cliff.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
BIN="${RCCL_TESTS_DIR:-$BENCH_ROOT/rccl-tests/src/build}"
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rccl/rccl_tests_$TS; mkdir -p "$OUT"
SUM=$OUT/rccl_tests_summary.txt
GPU_COUNTS="${GPU_COUNTS:-2 3 4 5 6 7 8}"
COLLECTIVES="${COLLECTIVES:-all_reduce all_gather}"
CONFIGS="${CONFIGS:-default tree ring no_mscll proto_simple}"

apply_config() {
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=0 \
         NCCL_SHM_DISABLE=0 NCCL_DEBUG=WARN
  export RCCL_MSCCL_ENABLE=1 NCCL_ALGO=Ring,Tree NCCL_PROTO=Simple,LL,LL128
  case "$1" in
    default)      ;;
    tree)         export NCCL_ALGO=Tree ;;
    ring)         export NCCL_ALGO=Ring ;;
    no_mscll)     export RCCL_MSCCL_ENABLE=0 ;;
    proto_simple) export NCCL_PROTO=Simple ;;
  esac
}

{ echo "RCCL config sweep $TS"; echo;
  printf '%-12s %-14s %3s %10s %14s\n' config collective N max_size busbw_GB/s; } | tee "$SUM"
assert_gpus_idle
for cfg in $CONFIGS; do
  apply_config "$cfg"
  for coll in $COLLECTIVES; do
    exe="$BIN/${coll}_perf"; [[ -x "$exe" ]] || continue
    for N in $GPU_COUNTS; do
      log="$OUT/${coll}_${cfg}_n${N}.log"
      timeout 900 "$exe" -b 16M -e 8G -f 2 -g "$N" -n 20 -w 5 -c 1 >"$log" 2>&1
      read -r size busbw < <(awk '/^ *[0-9]/ {sz=$1; bw=$(NF-1)} END {print sz, bw}' "$log")
      printf '%-12s %-14s %3s %10s %14s\n' "$cfg" "$coll" "$N" "${size:--}" "${busbw:--}" | tee -a "$SUM"
    done
  done
done
echo "results: $OUT"
```

### B.5 `rccl-tests/analyze_rccl.py` — NEW

Parses the raw logs directly (not the summary text), so it works on partial runs.

```python
#!/usr/bin/env python3
"""Parse rccl-tests logs into results/rccl.{md,csv} and flag the non-power-of-2 cliff.

Usage: analyze_rccl.py <log_dir> [<log_dir> ...] -o results
Handles both naming schemes: <coll>_n<N>.log and <coll>_<config>_n<N>.log
"""
import argparse, csv, math, re
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^(?P<coll>[a-z_]+?)(?:_(?P<cfg>default|tree|ring|no_mscll|proto_simple))?_n(?P<n>\d+)\.log$")

def parse(path: Path):
    """Return {size_bytes: busbw} from the in-place columns of a rccl-tests table."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < 8 or not f[0].isdigit():
            continue
        try:
            out[int(f[0])] = float(f[-2])   # in-place busbw is second-to-last column
        except ValueError:
            continue
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in a.dirs:
        for log in sorted(d.glob("*.log")):
            m = NAME.match(log.name)
            if not m: continue
            pts = parse(log)
            if not pts: continue
            top = max(pts)
            rows.append({"run": d.name, "collective": m["coll"], "config": m["cfg"] or "default",
                         "gpus": int(m["n"]), "max_size_bytes": top,
                         "busbw_at_max_GBps": pts[top],
                         "peak_busbw_GBps": max(pts.values()),
                         "n_sizes": len(pts)})
    if not rows: raise SystemExit("no rccl logs parsed")
    with (a.out / "rccl.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def table(sel, key, title):
        sub = [r for r in rows if sel(r)]
        if not sub: return []
        ks = sorted({r[key] for r in sub}); ns = sorted({r["gpus"] for r in sub})
        idx = {(r[key], r["gpus"]): r["busbw_at_max_GBps"] for r in sub}
        L = [f"## {title}", "", f"| {key} | " + " | ".join(f"N={n}" for n in ns) + " | cliff |",
             "|" + "---|" * (len(ns) + 2)]
        for k in ks:
            vals = [idx.get((k, n)) for n in ns]
            cells = [f"{v:.1f}" if v else "-" for v in vals]
            # cliff = worst non-power-of-2 N vs the mean of its power-of-2 neighbours
            p2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) == 0]
            np2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) != 0]
            flag = "-"
            if p2 and np2:
                ratio = min(np2) / (sum(p2) / len(p2))
                flag = f"{(1-ratio)*100:.0f}% ↓" if ratio < 0.85 else "none"
            L.append(f"| {k} | " + " | ".join(cells) + f" | {flag} |")
        return L + [""]

    L = ["# RCCL collective bandwidth — MI355X ×8, XGMI (busbw at top message size)", "",
         "Built natively for gfx950 against host ROCm 7.14 (no `HSA_OVERRIDE_GFX_VERSION`), "
         "so absolute numbers may exceed the Dell Cloud gfx942-override run.", ""]
    L += table(lambda r: r["config"] == "default", "collective", "All collectives (default config)")
    for coll in sorted({r["collective"] for r in rows if r["config"] != "default"}):
        L += table(lambda r, c=coll: r["collective"] == c, "config", f"`{coll}` — config comparison")
    L += ["## How to read the cliff column", "",
          "`X% ↓` means the worst non-power-of-2 N is that much below the mean of the "
          "power-of-2 Ns. If the `default` config cliffs but `tree`/`ring`/`no_mscll` do not, "
          "the recovering knob is a one-env-var workaround. If nothing recovers it, the gap is "
          "missing RCCL tuning for those arities on gfx950 — and per `rccl-tests.md`, a clean "
          "RVS `pqt` run (Part A) is what lets you attribute it to the algorithm layer rather "
          "than to the fabric.", ""]
    (a.out / "rccl.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rccl.md and .csv ({len(rows)} rows)")

if __name__ == "__main__":
    main()
```

### B.6 `rccl-tests/plot_rccl_busbw.py` — NEW

The repo's version hardcodes its numbers in a `data = {...}` dict. Ours reads
`results/rccl.csv`, so the figure regenerates automatically:

```python
#!/usr/bin/env python3
"""Plot busbw vs N per collective from results/rccl.csv -> results/rccl_busbw.png"""
import csv, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/rccl.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/rccl_busbw.png"
series = defaultdict(dict)
for r in csv.DictReader(open(src)):
    if r["config"] != "default": continue
    series[r["collective"]][int(r["gpus"])] = float(r["busbw_at_max_GBps"])

fig, ax = plt.subplots(figsize=(9, 5.5))
for coll, pts in sorted(series.items()):
    ns = sorted(pts)
    ax.plot(ns, [pts[n] for n in ns], marker="o", label=coll)
for n in (2, 4, 8):
    ax.axvline(n, color="0.85", lw=1, zorder=0)
ax.set(xlabel="GPUs (N)", ylabel="busbw (GB/s)",
       title="RCCL busbw at top message size — MI355X ×8, XGMI")
ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(dst, dpi=150)
print("wrote", dst)
```

### B.7 Part B run order

```bash
source common/env.sh; cd rccl-tests
COLLECTIVES=all_reduce GPU_COUNTS=8 ./run-rccl-all.sh              # ~2 min smoke
nohup ./run-rccl-all.sh     > $LOG_ROOT/rccl/all.out     2>&1 &    # 45–90 min
nohup ./run-rccl-configs.sh > $LOG_ROOT/rccl/configs.out 2>&1 &    # 30–45 min (sequential!)
$PY analyze_rccl.py $LOG_ROOT/rccl/rccl_all_* $LOG_ROOT/rccl/rccl_tests_* -o $BENCH_ROOT/results
$PY plot_rccl_busbw.py $BENCH_ROOT/results/rccl.csv $BENCH_ROOT/results/rccl_busbw.png
```

---

## PART C — Primus (including Megatron-LM)

Reproduces `primus/REPORT.md`. **This is where Megatron-LM is benchmarked** — not in Part B.

### C.1 Setup

```bash
cd $BENCH_ROOT/primus
git clone https://github.com/AMD-AIG-AIMA/Primus.git
docker pull rocm/primus:v26.5
df -h /                                   # confirm headroom after the pull
docker run --rm rocm/primus:v26.5 bash -c 'python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"'
```

That last command is the **go/no-go gate**: the arch list must contain `gfx950`. If it does
not, fall back to `docker pull rocm/primus:v25.9_gfx950` and use it for Megatron (the Dell Cloud
proven path), keeping v26.5 for the microbenches.

> ✅ **Gate passed 2026-08-13** (see §1.3). Also probed the image's actual CLI surface:
> all five subcommands the sweep uses exist —
> `{gemm, attention, gemm-dense, gemm-deepseek, strided-allgather, rccl}`. `strided-allgather`
> is new and unused here.
>
> ⚠️ **Corrected: the Megatron `EXP` path in §C.4 does not exist.** The plan's
> `examples/megatron/configs/llama2_7B-pretrain.yaml` is gone — configs are now per-arch.
> The correct path, present in both the image and upstream HEAD, is
> **`examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml`** — which is better than
> what the plan asked for, since it is MI355X-tuned. `primus/configs/models/megatron/llama2_7B.yaml`
> (the tokenizer-sed target) *does* exist, unchanged, so the offline-tokenizer trick stands.
>
> ⚠️ **Corrected: do not bind-mount the host clone at `/workspace`.** The plan's §C.2/§C.3
> scripts mount `$PRIMUS:/workspace`, but the host clone (HEAD `abc46648`) has drifted from
> the image's pinned tree (`b511d1b6`) — that is precisely the API-drift failure the next
> bullet warns about, reintroduced by the mount. The committed scripts instead run the
> image's own `/workspace/Primus` and bind-mount **only an output dir at `/out`**.

Two rules carried over from dell-cloud's troubleshooting notes, both of which cost real runs:

- **Do not bind-mount the cloned `Primus/` over the image's `/workspace/Primus`.** Its API is
  pinned to a specific `primus_turbo`; HEAD has drifted and `PrimusTurboAttention` fails to
  import. The host clone is for reading configs only — **not** a mount source. (Applies to
  v26.5 as well as `_gfx950`; see the correction above.)
- **Drop `--op all_reduce` from the rccl bench.** Primus' argparse advertises `all_reduce`
  but the backend expects `allreduce`; omitting the flag uses the correct default.

### C.2 `primus/run_gpu_scan.sh` — NEW (docker port)

```bash
#!/usr/bin/env bash
# Quick GEMM-only scan 1..8 GPUs (~5 min) — smoke test before the full sweep.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"
PRIMUS=$BENCH_ROOT/primus/Primus
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/gpu-scan-$RUN_ID; mkdir -p "$BASE"
OUT_HOST=$PRIMUS/sweep_out_$RUN_ID; mkdir -p "$OUT_HOST"
SUM=$BASE/summary.txt
assert_gpus_idle
echo "Primus GPU scan $RUN_ID (image $IMG)" | tee "$SUM"
for N in 1 2 3 4 5 6 7 8; do
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  log=$BASE/scan_${N}gpu.log; start=$(date +%s)
  echo "----- gemm N=$N devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout 600 docker run $(dgpu_args) \
      -v "$PRIMUS":/workspace -w /workspace \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      "$IMG" bash -c \
      "./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10 \
         --output-file /workspace/sweep_out_$RUN_ID/gemm_N${N}.md" >"$log" 2>&1
  rc=$?; echo "  rc=$rc duration=$(($(date +%s)-start))s log=$log" | tee -a "$SUM"
done
echo "$RUN_ID" > $LOG_ROOT/primus/CURRENT_RUN_ID.txt
```

### C.3 `primus/run_full_sweep.sh` — NEW (docker port)

Same bench matrix as the dell-cloud script, minus the Megatron entry — Megatron gets its own
script so a Megatron failure doesn't cost the whole hour of microbenches.

```bash
#!/usr/bin/env bash
# Primus microbench sweep: gemm, gemm-dense, gemm-deepseek, attention, rccl × N=1..8 (~1 h).
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"
PRIMUS=$BENCH_ROOT/primus/Primus
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/sweep-$RUN_ID; mkdir -p "$BASE"
OUT_HOST=$PRIMUS/sweep_out_$RUN_ID; mkdir -p "$OUT_HOST"
OUT_CTR=/workspace/sweep_out_$RUN_ID
SUM=$BASE/summary.txt
assert_gpus_idle

{ echo "Primus full sweep $RUN_ID"; echo "Image      : $IMG"; echo "Repo       : $PRIMUS";
  echo "Driver log : $BASE"; echo "Bench out  : $OUT_HOST";
  echo "Started    : $(date -Iseconds)"; echo; } | tee "$SUM"

run() {
  local name=$1 N=$2 t=$3 cmd=$4 devs port log start rc dur status
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  log="$BASE/${name}_N${N}.log"; start=$(date +%s)
  echo "----- $name N=$N port=$port devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout --signal=TERM --kill-after=30s "$t" \
    docker run $(dgpu_args) -v "$PRIMUS":/workspace -w /workspace \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
      "$IMG" bash -c "$cmd" >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  case $rc in 0) status=OK ;; 124) status="TIMEOUT(${t}s)" ;; *) status="FAIL(rc=$rc)" ;; esac
  echo "  $status duration=${dur}s log=$log" | tee -a "$SUM"
}

for N in 1 2 3 4 5 6 7 8; do
  echo "================ N=$N ================" | tee -a "$SUM"
  run gemm          $N  300 "./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10 --output-file $OUT_CTR/gemm_N${N}.md"
  run gemm-dense    $N  600 "./primus-cli direct -- benchmark gemm-dense --duration 5 --output-file $OUT_CTR/gemm-dense_N${N}.md"
  run gemm-deepseek $N  600 "./primus-cli direct -- benchmark gemm-deepseek --duration 5 --output-file $OUT_CTR/gemm-deepseek_N${N}.md"
  run attention     $N 1200 "./primus-cli direct -- benchmark attention --backend flash --mbs-list 4 --report-csv-path $OUT_CTR/attention_N${N}.csv"
  if (( N >= 2 )); then
    # NB: no --op flag (argparse says all_reduce, backend wants allreduce)
    run rccl        $N  900 "./primus-cli direct -- benchmark rccl --output-file $OUT_CTR/rccl_N${N}.md"
  else
    echo "----- rccl N=1 SKIPPED (collective needs N>=2) -----" | tee -a "$SUM"
  fi
done
echo; echo "Finished   : $(date -Iseconds)" | tee -a "$SUM"
```

### C.4 `primus/run_megatron.sh` — NEW (docker port, GBS fixed up front)

The repo needed **three** scripts here (`rerun_megatron_gfx950.sh`,
`_v2.sh`, `_missing.sh`) because its fixed `global_batch_size=256` is not divisible by
`MBS(4) × DP(N)` for N ∈ {3,5,6,7}, so half the sweep failed and had to be re-run with
per-N batch sizes. Compute a valid GBS per N in the first place and it is one script:

```
GBS(N) = MBS × N × GRAD_ACC     # always divisible by MBS×N, by construction
       = 4 × N × 8 = 32N        # weak scaling: constant work per GPU
```

```bash
#!/usr/bin/env bash
# Megatron-LM llama2-7B BF16 pretrain via Primus, N=1..8 (the headline benchmark).
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"          # fallback: rocm/primus:v25.9_gfx950
RUN_ID="${1:-$(cat $LOG_ROOT/primus/CURRENT_RUN_ID.txt 2>/dev/null || date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/sweep-$RUN_ID; mkdir -p "$BASE"
SUM=$BASE/summary.txt
MBS="${MBS:-4}"; GRAD_ACC="${GRAD_ACC:-8}"; TIMEOUT="${TIMEOUT:-3600}"
assert_gpus_idle
echo "================ MEGATRON $(date -Iseconds) image=$IMG ================" | tee -a "$SUM"

run_megatron() {
  local N=$1 devs port log start rc dur status GBS
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  GBS=$(( MBS * N * GRAD_ACC ))
  log="$BASE/megatron-llama2_7B-bf16_N${N}.log"; start=$(date +%s)
  echo "----- megatron N=$N GBS=$GBS devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout --signal=TERM --kill-after=30s "$TIMEOUT" \
    docker run $(dgpu_args) -w /workspace/Primus \
      -e EXP="${EXP:-examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml}" \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
      "$IMG" bash -c '
        MY=/workspace/Primus/primus/configs/models/megatron/llama2_7B.yaml
        # offline tokenizer: no HF download, mock_data works
        sed -i "s|tokenizer_type: Llama2Tokenizer|tokenizer_type: NullTokenizer|" "$MY"
        sed -i "/^tokenizer_model:/d" "$MY"
        grep -q "^vocab_size:" "$MY" || echo "vocab_size: 32000" >> "$MY"
        bash examples/run_pretrain.sh global_batch_size='"$GBS"' micro_batch_size='"$MBS"'
      ' >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  case $rc in 0) status=OK ;; 124) status="TIMEOUT(${TIMEOUT}s)" ;; *) status="FAIL(rc=$rc)" ;; esac
  echo "  $status duration=${dur}s log=$log" | tee -a "$SUM"
}

for N in 1 2 3 4 5 6 7 8; do run_megatron "$N"; done
echo "[megatron] $(date -Iseconds) DONE" | tee -a "$SUM"
```

Notes carried from the Dell Cloud experience:

- **First run pays a ~10 min JIT tax** (aiter / triton / primus-turbo compile hundreds of MB
  of `.cuda.o`). Subsequent N reuse it — because `$CACHE_ROOT/triton` is bind-mounted, the
  cache survives container exit. This is the docker equivalent of dell-cloud's 20 GiB overlay,
  and it is why no overlay file is needed.
- **N=1 is the slowest** (~42 s/iter → 50 iters ≈ 45 min including JIT). Timeout is raised to
  3600 s from dell-cloud's 1800 s so N=1 completes; steady-state TF/s is reliable from the
  first ~10 captured iters even if it is cut short.

### C.5 Analysis — reuse `generate_report.py`

Copy `../dell-cloud/primus/generate_report.py` unmodified. It parses Megatron
`throughput per GPU` lines, the GEMM family markdown tables, attention CSVs, and RCCL
`eff_gbps`, then writes `REPORT.md` with TF/s-vs-N tables plus a computed analysis section.

```bash
$PY primus/generate_report.py \
    $LOG_ROOT/primus/sweep-$RUN_ID \
    $BENCH_ROOT/primus/Primus/sweep_out_$RUN_ID \
    $REF_ROOT/megatron-lm/summary.md \
    $BENCH_ROOT/results/PRIMUS_REPORT.md
```

Arg 3 is the B200 comparison table source — the Dell Cloud suite's own
`megatron-lm/summary.md` contains a `| N | B200 TF/s/GPU | MI355X TF/s/GPU |` block, which
gets lifted verbatim into §1.2. Pass `/dev/null` to omit that section.

### C.6 Part C run order

```bash
source common/env.sh; cd primus
RUN_ID=$(date +%Y%m%d-%H%M%S)
./run_gpu_scan.sh "$RUN_ID"                                              # ~5 min gate
nohup ./run_full_sweep.sh "$RUN_ID" > $LOG_ROOT/primus/sweep.out 2>&1 &  # ~1 h
nohup ./run_megatron.sh   "$RUN_ID" > $LOG_ROOT/primus/mega.out  2>&1 &  # 1.5–3 h (AFTER the above)
$PY generate_report.py $LOG_ROOT/primus/sweep-$RUN_ID \
    $BENCH_ROOT/primus/Primus/sweep_out_$RUN_ID \
    $REF_ROOT/megatron-lm/summary.md \
    $BENCH_ROOT/results/PRIMUS_REPORT.md
```

---

## 5. Execution order and wall-clock budget

Everything is **strictly sequential** — all three suites want all 8 GPUs and the whole
11.2 kW OAM power envelope.

| Step | Task | Est. |
|------|------|------|
| 0 | ~~apt dev packages~~ (already installed), venv, clones, `docker pull` | ✅ **done** |
| 1 | Build RVS + rccl-tests | ✅ **done** |
| 2 | **A** — RVS smoke → gst sweep → health modules | ~1 h |
| 3 | **B** — RCCL all-collective sweep | 45–90 min |
| 4 | **B** — RCCL config sweep (5 configs) | 30–45 min |
| 5 | **C** — Primus GPU scan (gate) | 5 min |
| 6 | **C** — Primus microbench sweep | ~1 h |
| 7 | **C** — Megatron llama2-7B N=1..8 | 1.5–3 h |
| 8 | Analysis + report assembly | 20 min |
| | **Total** | **6–9 h** |

Long steps run under `nohup` and are polled, so the session is not blocked.

## 6. Deliverables in `results/`

| File | From |
|------|------|
| `rvs_tflops.md` / `.csv` | `analyze_rvs.py` — TFLOPS × 9 precisions × N=1..8, % of peak, scaling |
| `rvs_health.md` | `run_rvs_health.sh` summary, hand-checked pass/fail per module |
| `rccl.md` / `.csv` | `analyze_rccl.py` — busbw per collective × N, config comparison, cliff flags |
| `rccl_busbw.png` | `plot_rccl_busbw.py` |
| `PRIMUS_REPORT.md` | `generate_report.py` — GEMM / attention / RCCL / **Megatron TF/s/GPU × N** |
| `SUMMARY.md` | Hand-written: cross-suite findings + deltas vs the Dell Cloud published numbers |

## 7. Risks and open items

1. **Disk on `/` (216 GB).** The single largest risk. A foreign stopped container holds
   283 GB of reclaimable layers. Mitigation: one image at a time, check `df -h /` between
   pulls, all logs/caches on `/mnt/scratch`. Moving docker's `data-root` to `/mnt/scratch`
   would remove the risk entirely but restarts dockerd — **needs your approval.**
2. **`rocm/primus:v26.5` is unvalidated for this workload.** Dell Cloud used v26.3 + v25.9_gfx950.
   v26.5 matches the host ROCm exactly, which is why it is the first choice; C.1's
   `torch.cuda.get_arch_list()` check is the gate, and `v25.9_gfx950` is the fallback.
3. **This node runs a k8s cluster** (cilium interfaces, PVCs on `/mnt/disk0`). If pods are
   scheduled onto these GPUs, results will be noise. `assert_gpus_idle` warns; check
   `rocm-smi` before each long step.
4. **Numbers will not match the Dell Cloud results exactly**, and shouldn't: newer ROCm (7.14 vs
   6.4/7.0), native gfx950 instead of the gfx942 override, docker instead of singularity, and
   a different host. Expect RCCL and GEMM to come out equal-or-better; treat any regression
   as a finding worth chasing.
5. **RVS `gst` peaks are power-limited at N=8** on the dense precisions — the Dell Cloud notes record the
   11.2 kW tray throttles. A per-GPU TFLOPS drop from N=1 to N=8 is expected physics, not a bug.
6. **`alltoallv` OOM at N=5** killed the reference sweep mid-run. Capped at 4 GiB via
   `ALLTOALL_MAX`; if it still OOMs, drop to 2 GiB — the plateau is reached well before then.

---

## 8. Setup log (2026-08-13)

Steps 0 and 1 executed. **No benchmark has been run** — the box is staged and idle,
waiting on the go-ahead.

### What was done

| # | Action | Result |
|---|--------|--------|
| 0.1 | Host dev packages | **No-op** — all 5 already installed at `7.14.0-3`; headers + `hipcc` verified |
| 0.2 | `venv` at `/mnt/scratch/shaohao/venv` | Python 3.10.12 + matplotlib, pandas, tabulate ✔ |
| 0.3 | Clone RVS / rccl-tests / Primus | 17 MB / 772 K / 163 MB ✔ |
| 0.4 | `docker pull rocm/primus:v26.5` | 54.8 GB; gfx950 gate **passed**, fallback image not needed |
| 0.5 | `common/env.sh`, cache dirs, log dirs | sources clean; `RCCL_TESTS_DIR` added to the plan's version |
| 1.1 | Build + install RVS | all **15** modules incl. `pebb`/`pbqt`/`pulse`; `rvs -g` → 8 MI355X ✔ |
| 1.2 | Build rccl-tests | **12** `*_perf` binaries, native gfx950 ✔ |
| 1.3 | Write all 13 scripts | `bash -n` + `py_compile` clean on every one |

### Five places reality differed from the plan

1. **The dev-headers premise was false** (§1.1). Everything was already installed; no `apt`
   run was needed. §0's "runtime only, dev headers missing" row is wrong.
2. **RVS needs `git submodule update --init --recursive`** (§A.1) or cmake hard-fails on
   TransferBench. The plan omitted it.
3. **RVS has no `pqt` module** (§A.4). The health script's P2P/XGMI entry pointed at a
   nonexistent module — the one test that Part B's cliff attribution rests on. Now `pbqt`.
4. **The Megatron `EXP` config path no longer exists** (§C.1/§C.4). Now the MI355X-specific
   `examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml`.
5. **The Primus microbench scripts must not mount the host clone over `/workspace`**
   (§C.2/§C.3) — it reintroduces the exact API drift the plan warns about two paragraphs
   later. Output-only mount at `/out` instead.

Items 3–5 were plan bugs that would each have cost a run or silently voided a conclusion.

### Disk

`/` is at **161 G free** (was 214 G) after the 54.8 GB image. Since `v25.9_gfx950` is not
needed, no further large pull is planned, and the risk in §7.1 is largely retired. The
foreign 283 GB stopped container was **not** touched.

### Not yet done — deliberately

- No smoke tests run (they are GPU workloads; §A.6/§B.7/§C.6 start them).
- Untested at runtime: the Primus microbench **flag surface** (`--duration`, `--mbs-list`,
  `--report-csv-path`) is carried over from the older v26.3/v25.9 images and has not been
  validated against v26.5. `run_gpu_scan.sh` is the 5-minute gate that will catch any drift
  before the 1-hour sweep commits to it.
- `results/` is empty; nothing is committed to git yet.

### To start, on your go

```bash
cd /home/amd/shaohao/amd-benchmarks/amd-cloud && source common/env.sh
cd work-rocmval && GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 ./run_tflops.sh
```
