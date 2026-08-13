# amd-benchmarks

Benchmarks for AMD Instinct GPUs, organized by the machine they were run on.

Each top-level directory is one physical server. Results are **not** comparable across
directories without care — the servers differ in ROCm version, container runtime, and
GPU generation, and each suite records its own environment.

## Servers

| Directory | Server | GPUs | Status |
|-----------|--------|------|--------|
| [`dell-cloud/`](dell-cloud/) | **Dell Cloud** server | 8 × MI355X (gfx950) | Complete — original results |
| [`amd-cloud/`](amd-cloud/) | **AMD Cloud** server | 8 × MI355X (gfx950) | In progress |

### [`dell-cloud/`](dell-cloud/) — Dell Cloud server

All of the original benchmark work in this repository. **These runs were performed on a
Dell Cloud server.** RHEL-family host, Singularity/Apptainer containers, ROCm 6.4.3–7.0.

- [`work-rocmval/`](dell-cloud/work-rocmval/) — ROCm Validation Suite (`rvs`) TFLOPS tests,
  run scripts, notes, and sample output under `tflops_runs/`.
- [`megatron-lm/`](dell-cloud/megatron-lm/) — Megatron-LM training benchmark scripts,
  notes, and TF/s sweep results.
- [`rccl-tests/`](dell-cloud/rccl-tests/) — RCCL collective-bandwidth sweeps and the
  N=5/6/7 cliff analysis.
- [`primus/`](dell-cloud/primus/) — Primus sweep + report harness (GEMM, attention, RCCL,
  Megatron-LM llama2-7B) and the generated `REPORT.md`.

### [`amd-cloud/`](amd-cloud/) — AMD Cloud server

Benchmarking on the server provided by **AMD Cloud**. Ubuntu 22.04 host, Docker containers,
ROCm 7.14 with native gfx950 support. Everything for this server — plans, run scripts,
analysis scripts, logs, and results — lives under `amd-cloud/`. See
[`amd-cloud/README.md`](amd-cloud/README.md) for the layout and
[`amd-cloud/plan.md`](amd-cloud/plan.md) for the full benchmark plan.

## What this repo tracks

**Text only** — scripts, plans, markdown, CSV, and benchmark logs. Everything bulky stays
off-repo, by convention under `/mnt/scratch/shaohao/` on the benchmark host:

| Kind | Example | Where it lives |
|---|---|---|
| Model weights / checkpoints | `Qwen3-8B-FP8` (8.9 GB), large MoE (up to 1.5 TB) | `/mnt/scratch/shaohao/models/` |
| Upstream source + build trees | ROCmValidationSuite, rccl-tests, Primus, ATOM | cloned in place, git-ignored |
| Container layers, JIT caches | Docker, Triton, HF, pip | `/var/lib/docker`, `/mnt/scratch/shaohao/cache/` |
| Profiler traces | torch/kineto `*.trace.json` | not collected by default; ignored if produced |

Two mechanisms enforce this:

1. **[`.gitignore`](.gitignore)** — repo-wide patterns for weights, archives, container
   images, traces, and build output. Per-directory rules (the upstream clones) live in
   [`amd-cloud/.gitignore`](amd-cloud/.gitignore).
2. **A `pre-commit` hook** with two guards, because a pattern list only catches what you
   thought to name:
   - **per-file**, blocks any staged file over 20 MB (`MAX_MB`)
   - **per-commit**, blocks a commit adding more than 100 MB in total (`TOTAL_MB`) — a run
     directory of many medium logs slips under the per-file bar but still bloats history

   GitHub warns above 50 MB and hard-rejects above 100 MB, so this fails locally instead of
   at push time.

Git does not clone hooks, so **run this once in a fresh clone**:

```bash
bash .githooks/install.sh     # sets core.hooksPath=.githooks
```

Override the threshold with `MAX_MB=50 git commit …`, or bypass deliberately with
`git commit --no-verify`.
