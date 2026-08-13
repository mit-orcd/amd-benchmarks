# Dell Cloud server

**All benchmarks in this directory were run on a Dell Cloud server.**

This is the original body of work in the repository; it was moved here unchanged when a
second machine (see [`../amd-cloud/`](../amd-cloud/)) was added, so every path, script, and
number below refers to the Dell Cloud host.

## Environment

As documented by the suites themselves (`work-rocmval/readme.md`,
`megatron-lm/readme-rccl.md`, `primus/README.md`):

| Item | Value |
|------|-------|
| GPUs | 8 × AMD Instinct MI355X (gfx950) |
| Interconnect | XGMI all-to-all, 1 hop between every pair, equal weight; GPU 0–3 → NUMA 0, GPU 4–7 → NUMA 1 |
| OS | RHEL-family (default GCC 8.x toolchain) |
| Container runtime | Singularity/Apptainer ≥ 1.4, plus rootless podman for image pulls |
| ROCm | 6.4.3 / 7.0-era container images (`rocm/megatron-lm`, `rocm/primus`) |
| Python | 3.11 at `/usr/bin/python3.11` |
| Working root | `/home/v89592/shaohao/` — paths are hard-coded in most scripts |

Two consequences of that environment show up throughout the scripts and are **specific to
this machine**:

- The RVS build disables the `pebb`, `pbqt`, and `pulse` modules, because
  `TransferBench.hpp` needs C++20 `<barrier>` (libstdc++ ≥ 11) and the host only had GCC 8.
- `HSA_OVERRIDE_GFX_VERSION=9.4.2` is set almost everywhere, aliasing gfx950 → gfx942,
  because the container images of that era shipped no gfx950 code objects.

## Contents

- [`work-rocmval/`](work-rocmval/) — ROCm Validation Suite (`rvs`) TFLOPS tests, including
  run scripts, notes, and sample run output under `tflops_runs/`.
- [`megatron-lm/`](megatron-lm/) — Megatron-LM training benchmark scripts, notes, and TF/s
  sweep results (`run.sh`, `summary*.md`, `tflops-gap-analysis.md`).
- [`rccl-tests/`](rccl-tests/) — RCCL collective-bandwidth sweeps and the analysis showing
  the N=5/6/7 cliff lives in RCCL rather than in Megatron-LM. Originally filed under
  `megatron-lm/`; split out because the subject is RCCL, not training.
- [`primus/`](primus/) — Primus sweep + report harness across GEMM, attention, RCCL, and
  Megatron-LM llama2-7B pretraining, with the generated `REPORT.md`.

`megatron-lm/` and `rccl-tests/` were run from the same on-host working directory
(`/home/v89592/shaohao/megatron-lm/work/`) in the same container, so paths inside those
documents still refer to the original shared location.
