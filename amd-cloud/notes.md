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

## Why Part D (ATOM) uses tensor parallelism, not PP/DP/EP

For single-node inference serving, TP is the right — and largely only sensible — choice:

- **PP** (pipeline parallel) pipelines layers across GPUs, adding bubble overhead. It only
  pays off across nodes or when a model won't fit otherwise. On one node with fast XGMI,
  it's strictly worse for latency than TP.
- **DP** (data parallel) replicates the whole model per GPU — impossible for the 70B/Kimi-K3
  tiers (weights alone don't fit on one GPU), and for the 8B tier it would just measure 8
  independent servers, not one scaled server.
- **EP** (expert parallel) only applies to MoE layers, and it's a *complement* to TP, not a
  replacement — ATOM uses MoE-specific paths for Kimi-K3 under the hood, alongside TP.

TP shards every layer's weights across GPUs and all-reduces activations each layer, which is
both the standard serving configuration and the reason Part D matters as a benchmark: it puts
**RCCL in the per-token critical path**, connecting Part D's latency directly to Part B's
collective results (see [[amd-cloud-plan]] for the RCCL non-power-of-2 cliff finding).

Tier 1 (Qwen3-8B) runs at **TP=1 deliberately** — single GPU, no collectives at all — as the
control, so tiers 2 (Llama-70B) and 3 (Kimi-K3) at **TP=8** isolate what tensor parallelism
costs relative to that baseline.
