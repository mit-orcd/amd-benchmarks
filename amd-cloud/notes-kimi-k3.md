# Kimi-K3 rerun plan — AMD's official MAD recipe

Status: **not run yet.** This is a plan for a follow-up run, to be triggered later on request.
The existing tier-3 result (`results/atom.md`, `results/kimi-k3-base.md`) stays as-is until then —
this file does not retroactively invalidate it, but flags it may not reflect AMD's best
validated config.

## What prompted this

Fetched `https://raw.githubusercontent.com/ROCm/MAD/develop/benchmark/kimi_k3/README.md`
(AMD's own MAD benchmark harness repo) and compared it against what we actually ran.

## How the original run was prepared

Not written from scratch. Model flags (TP=8, `--online_quant_config` PTPC-FP8,
`--no-enable_prefix_caching`, etc.) came from ATOM's in-repo `ATOM/recipes/Kimi-K3.md`.
Everything else — server start/stop with safety guards (refuse rather than pkill on a
shared box), the concurrency-sweep driver, the Part D orchestrator, and the
analysis/report generators — was written for this project.

## What MAD does differently

### 1. Image

| | Used | MAD recommends |
|---|---|---|
| Tag | `rocm/atom-dev:latest` (generic) | `rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3` (Kimi-K3-specific, dated build) |

### 2. Launch env vars — MAD sets, we did not

```
ATOM_LOADER_USE_THREADPOOL=1
ATOM_LOADER_THREADPOOL_WORKERS=16
ATOM_SYNC_AFTER_LOAD=1
ATOM_DIST_TIMEOUT_SECONDS=3600
ATOM_USE_TRITON_GEMM=1
AITER_USE_GROUPED_GEMM=0
ATOM_USE_TRITON_MOE=0
AITER_FLYDSL_FORCE=1
AITER_FORCE_GFX1250=0
ATOM_USE_UNIFIED_ATTN=1
ATOM_FORCE_ATTN_TRITON=1
```

### 3. Server flags — MAD's ATOM launch command

```bash
python -m atom.entrypoints.openai_server \
  --model /model_weights --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code --max-model-len 16384 \
  --max-num-seqs 64 --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.93 --block-size 128 \
  --no-enable_prefix_caching
```

Differences from what we ran: `--max-num-batched-tokens 10240` (we used 16384), and **no
`--online_quant_config`** — MAD relies on the env-var kernel selection above instead of
explicit PTPC-FP8 quant config.

### 4. Benchmark client — MAD's standalone command

```bash
python -m atom.benchmarks.benchmark_serving \
  --model /model_weights --backend vllm \
  --base-url http://localhost:8000 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --dataset-name random --ignore-eos \
  --request-rate inf --random-range-ratio 0.8 \
  --trust-remote-code --max-concurrency 64 \
  --num-prompts 640 --random-input-len 1024 \
  --random-output-len 1024 --save-result \
  --result-dir ./ --result-filename kimi_k3_atom_serving.json
```

This matches what we ran closely: `--ignore-eos`, `--random-range-ratio 0.8`,
`--num-prompts` = 10x concurrency, ISL/OSL 1024/1024. No `--served-model-name` on the
server side, so client and server model IDs match automatically — consistent with the fix
already applied to `run_atom_server.sh` after the first (broken) Part D run.

MAD's own table says the **official sweep goes to concurrency 256** (64/128/256), even
though `--max-num-seqs 64` is set on the server — meaning concurrency above 64 is expected
to queue, not to be rejected. Our run capped `KIMI_CONC` at 64 to match `max-num-seqs`; MAD
runs past it deliberately to characterize queueing behavior.

## Correction to earlier guidance

I previously told the user that `ATOM_USE_UNIFIED_ATTN=1` / `ATOM_USE_TRITON_GEMM=1` were
gfx1201-consumer-only flags to avoid on MI355X (based on `ATOM/recipes/Qwen3-8B-FP8.md`,
which targets RX 9070 XT). **That guidance does not hold for Kimi-K3.** AMD's own MAD
harness, specifically targeting MI350X/MI355X, sets these same-named flags for this model.
Do not reapply the old "avoid Triton flags on MI355X" reasoning to a Kimi-K3 rerun.

## Why this matters

The pinned, dated, Kimi-K3-specific image is more likely to match the kernel set AMD
validated and published numbers against. The current `results/kimi-k3-base.md` bottleneck
analysis (HBM ~29% of roofline, EP unsupported via `NotImplementedError`) was measured
against the generic image + ATOM in-repo recipe — plausibly not AMD's best validated
config for this model. A MAD-config rerun is the way to check whether the bottleneck
picture changes.

## Rerun plan (when requested)

1. `docker pull rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3`
   — check `/` free space first (was ~123 GB free as of 2026-08-17; this image's size is
   unknown, confirm before pulling).
2. Update `atom/run_part_d.sh` tier-3 launch (or a new `run_kimi_mad.sh`, keeping the
   existing tier-3 result untouched for comparison) to:
   - use the MAD image tag instead of `rocm/atom-dev:latest`
   - add the 11 env vars listed above
   - drop `--online_quant_config`, use `--max-num-batched-tokens 10240`
3. Run the benchmark client with `KIMI_CONC` extended to `"64 128 256"` to match MAD's
   official sweep (server still has `--max-num-seqs 64`, so >64 will show queueing —
   expected, not a bug).
4. Compare against the existing `results/kimi-k3-base.md` / `results/atom.md` tier-3 numbers.
   If throughput improves meaningfully, the HBM-utilization analysis in `kimi-k3-base.md` §3
   should be re-derived against the new numbers, not just appended.
5. Keep both results if they differ substantially — label old vs new by image tag, don't
   silently overwrite.

---

# Rerun attempt 1 — 2026-08-18 — FAILED (`ModuleNotFoundError: No module named 'fla'`)

Launched `atom/run_kimi_mad.sh` with the MAD recipe. Image pulled OK, 1.5 TB model loaded
(~6.5 min), server reported READY — then **the first benchmark request killed the server**.
No summary was written (the "0 requests completed" guard held), so
`results/kimi-k3-mad.md` does not exist and no existing file was modified.

Run dir: `logs/atom/kimi_mad_20260818_183450/`

```
File "/app/ATOM/atom/models/kimi_k3.py", line 749, in _run_kda
    from fla.ops.kda import chunk_kda
ModuleNotFoundError: No module named 'fla'
[atom] AsyncIOProcManager(ModelRunner): [ModelRunner1/8] proc died unexpectedly (exitcode=1)
```

## What `fla` is and why Kimi-K3 needs it

**`fla` = `flash-linear-attention`** (PyPI, v0.5.2) — a third-party library of Triton kernels
for *linear-attention* architectures (Gated Linear Attention, DeltaNet, Mamba-2, RetNet, and
**KDA**). Not an AMD or ATOM component.

Kimi-K3 is a hybrid: **24 MLA full-attention + 69 KDA (Kimi Delta Attention) linear-attention
layers** out of 93. `fla` supplies `chunk_kda`, the chunked kernel computing those 69 KDA
layers — the majority of the model's attention. Without it the model cannot run its own
architecture.

**It is not optional and not flag-gated.** Verified by reading the image's own source:

- `kimi_k3.py:749` — `from fla.ops.kda import chunk_kda` sits **unconditionally at the top of
  `_run_kda`**; no flag guard, no try/except fallback.
- `kimi_k3.py:778` — `return chunk_kda(...)` is the only implementation; grep found no
  alternative KDA path.
- `kimi_k3.py:647` calls it "**the fla prefill path**"; a separate fused aiter kernel handles
  *decode* (`:783`).

So `fla` is required for **prefill** on every KDA layer — which is why the server passed load
and health checks, then died the instant real work arrived.

**Correction to an earlier hypothesis:** I initially guessed MAD's `ATOM_FORCE_ATTN_TRITON=1`
forced this Triton path. That was **wrong** — the import is unconditional. Dropping MAD env
vars would not avoid it.

An in-source comment shows the path is actively maintained and accuracy-tested: *"a bf16 beta
yields a bf16 write strength, which erodes the delta-rule state update across the 71 KDA
layers (measured gsm8k regression)"*.

## Why it is missing from AMD's own Kimi-K3 image

Reporting the fact, not the intent: `fla` is genuinely absent from
`rocm/atom-dev:rocm7.2.4_..._20260727_kimi_k3` — not pip-installed, not vendored anywhere on
the filesystem (checked both). Since ATOM's Kimi-K3 model file imports it unconditionally for
prefill, **the image as published cannot serve Kimi-K3**. That looks like a packaging
omission in this dated build.

Notably the earlier run on `rocm/atom-dev:latest` worked, so `latest` evidently *does* ship
it — this is a regression in the MAD-pinned tag.

## Fix (verified, not yet run end-to-end)

`pip install flash-linear-attention` inside the container. Tested with GPUs attached:

```
Successfully installed fla-core-0.5.2 flash-linear-attention-0.5.2
gpus 8 gfx950:sramecc+:xnack-
fla.ops.kda OK
triton 3.7.0
```

This adds a genuinely missing dependency; it is not a workaround for a config mistake.
(A "Triton is not supported on current platform, roll back to CPU" warning appears when the
container runs *without* GPU devices; with `--device /dev/kfd` attached it detects all 8
gfx950 GPUs correctly.)

---

# Fidelity to the MAD README — what matches and what does not

Checked line by line against
`https://raw.githubusercontent.com/ROCm/MAD/develop/benchmark/kimi_k3/README.md`.

## Server flags — exact match

All nine MAD flags identical: `--kv_cache_dtype fp8`, `-tp 8`, `--trust-remote-code`,
`--max-model-len 16384`, `--max-num-seqs 64`, `--max-num-batched-tokens 10240`,
`--gpu-memory-utilization 0.93`, `--block-size 128`, `--no-enable_prefix_caching`.
Added only `--server-port 8010` (MAD's example omits it, defaulting to 8000; changed to avoid
port collisions on this shared host).

## Env vars — all 11 present

Every MAD var is set. Additionally set `NCCL_IB_DISABLE=1`, `RCCL_MSCCL_ENABLE=1`,
`NCCL_DEBUG=WARN` — single-node RCCL settings carried over from Parts B–D of this project for
consistency.

## Client — matches, with one real gap

Identical on `--backend vllm`, `--percentile-metrics ttft,tpot,itl,e2el`,
`--dataset-name random`, `--ignore-eos`, `--request-rate inf`, `--random-range-ratio 0.8`,
`--trust-remote-code`, `--random-input-len/output-len 1024`, `--save-result`,
`--result-dir`/`--result-filename`, and `--num-prompts` = 10x concurrency (MAD shows 640 at
c=64; we compute `C*10` → 640/1280/2560).

**GAP: MAD's ATOM table specifies "input 1024 and 4096" — two input lengths. We sweep ISL=1024
only.** Adding ISL=4096 would be a second sweep dimension. Deliberate trade: ISL=1024 keeps
the result directly comparable with the original `kimi-k3-base.md` run. Decide before rerunning.

## Input files

None to tune — `--dataset-name random` synthesizes prompts, so there is no dataset to fetch.
Model weights point at the existing `/mnt/scratch/shaohao/models/Kimi-K3`, matching MAD's
`-v /path/to/Kimi-K3:/model_weights` pattern (mounted at `/model`, passed as `--model /model`).

## Deliberately not matched: madengine

MAD's primary path is `madengine run --tags pyt_atom_kimi-k3`, which installs their harness
and manages the container itself. We use their documented **"Standalone benchmarking"**
section instead, because it fits this repo's existing driver/logging/analysis structure —
madengine would want to manage model downloads and output layout its own way.

---

**Status: fix identified and verified in isolation; full rerun not yet re-launched.**
