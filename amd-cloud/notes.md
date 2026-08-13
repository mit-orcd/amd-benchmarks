# Notes — where things live on this host

## Container images

`/var/lib/docker` (Docker's default data-root, unchanged — not moved to `/mnt/scratch`,
since that would need a dockerd restart and was never approved).

| Image | Size | Origin |
|---|---|---|
| `rocm/atom-dev:latest` | 106 GB | pre-existing on host |
| `rocm/primus:v26.5` | 54.8 GB | pulled for Part C |

## Source code (git clones)

Inside `amd-cloud/`, one per part, all git-ignored so only our own scripts sync, not the
upstream trees:

| Part | Path |
|---|---|
| A — RVS | `amd-cloud/work-rocmval/ROCmValidationSuite/` |
| B — rccl-tests | `amd-cloud/rccl-tests/src/` |
| C — Primus | `amd-cloud/primus/Primus/` |
| D — ATOM | `amd-cloud/atom/ATOM/` |

## Executables (built from source)

Under those same clones, on `/` (the repo's filesystem, not `/mnt/scratch`):

| Binary | Path |
|---|---|
| `rvs` | `amd-cloud/work-rocmval/ROCmValidationSuite/install_local/bin/rvs` |
| rccl-tests `*_perf` (12 binaries) | `amd-cloud/rccl-tests/src/build/` |
| ATOM | not built — runs from inside the `rocm/atom-dev` image, nothing built on the host |

All of the above is on `/` (839 G total).

## Off-repo bulk (`/mnt/scratch/shaohao/`, separate 7 TB NVMe)

Only large/regenerable things go here — never `/`:

| What | Path |
|---|---|
| Model weights (Part D) | `models/` — Qwen3-8B-FP8 (8.9 GB), Llama-3.1-70B-Instruct-FP8 (68 GB), Kimi-K3 (1.5 TB) |
| Analysis venv | `venv/` |
| Container/JIT caches | `cache/{triton,hf,torch,pip}/` |
