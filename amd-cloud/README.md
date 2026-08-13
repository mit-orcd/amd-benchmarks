# AMD Cloud server

**All benchmarks in this directory run on the server provided by AMD Cloud.** Everything
for this machine — plans, run scripts, analysis scripts, logs, and results — lives here.
The prior work on a different machine is in [`../dell-cloud/`](../dell-cloud/), which this
directory treats as read-only reference: its scripts are the starting point and its numbers
are the comparison baseline, but nothing here writes into it.

Start with **[`plan.md`](plan.md)** — the full benchmark plan (what to install, what to
download, every run and analysis script, execution order, and risks).

## Environment

| Item | Value |
|------|-------|
| GPUs | 8 × AMD Instinct MI355X (gfx950) |
| CPU / RAM | 2 × AMD EPYC 9575F (256 threads) / 3.0 TiB |
| OS | Ubuntu 22.04.5, kernel 6.8.0-65 |
| amdgpu driver | 6.19.14.31400100 |
| ROCm | 7.14 (`/opt/rocm`) — native gfx950, installed runtime-only |
| Container runtime | Docker 29.7.2 (no singularity/apptainer/podman) |
| Interconnect | XGMI all-to-all, 1 hop between every GPU pair |
| Working root | `/home/amd/shaohao/amd-benchmarks/amd-cloud` |

Two differences from the Dell Cloud runs follow from that environment and apply to every
script here: containers are driven with **`docker run`** rather than `singularity exec`
(which also removes the need for the 20 GiB ext3 overlay), and
**`HSA_OVERRIDE_GFX_VERSION` is never set**, because ROCm 7.14 and the images we use are
gfx950-native — the gfx942 alias would undercount this hardware.

## Layout

| Path | Contents |
|------|----------|
| `plan.md` | The benchmark plan |
| `common/env.sh` | Shared paths, RCCL env, docker helper, GPU-idle check |
| `work-rocmval/` | **Part A** — ROCm Validation Suite: `gst` TFLOPS sweep + health modules |
| `rccl-tests/` | **Part B** — RCCL collective sweep. RCCL **only**; no Megatron here |
| `primus/` | **Part C** — Primus GEMM/attention/RCCL microbenches **and** Megatron-LM llama2-7B |
| `logs/{rvs,rccl,primus}/` | Per-run driver logs and summaries |
| `results/` | Final markdown / CSV / PNG deliverables |

Megatron-LM is benchmarked **only** through Primus (Part C). Part B mirrors
[`../dell-cloud/rccl-tests/`](../dell-cloud/rccl-tests/); the Megatron-LM training sweep in
`dell-cloud/megatron-lm/` is not reproduced.

Cloned upstream sources (ROCmValidationSuite, rccl-tests, Primus) are placed inside their
respective part directories but git-ignored — see [`.gitignore`](.gitignore). Bulk
regenerable caches (Docker layers, Triton/HF/pip JIT caches, the analysis venv) live at
`/mnt/scratch/shaohao/`, off-repo.

## Status

| Part | State |
|------|-------|
| A — ROCm Validation Suite | Planned |
| B — RCCL collectives | Planned |
| C — Primus + Megatron-LM | Planned |
