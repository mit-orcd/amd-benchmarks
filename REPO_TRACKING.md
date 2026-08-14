# What this repo tracks

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
