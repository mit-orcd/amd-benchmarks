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
| ROCm | 7.14 (`/opt/rocm`) — native gfx950, runtime **and** dev packages installed |
| Container runtime | Docker 29.7.2 (no singularity/apptainer/podman) |
| Interconnect | XGMI all-to-all, 1 hop between every GPU pair |
| Working root | `/home/amd/shaohao/amd-benchmarks/amd-cloud` |

Two differences from the Dell Cloud runs follow from that environment and apply to every
script here: containers are driven with **`docker run`** rather than `singularity exec`
(which also removes the need for the 20 GiB ext3 overlay), and
**`HSA_OVERRIDE_GFX_VERSION` is never set**, because ROCm 7.14 and the images we use are
gfx950-native — the gfx942 alias would undercount this hardware.

## Layout

```
amd-cloud/
├── plan.md                    the benchmark plan
├── notes.md                   host notes: where images/source/executables/weights live
├── common/env.sh              shared paths, RCCL env, docker helper, GPU-idle check
│
├── work-rocmval/              Part A — ROCm Validation Suite
│   ├── ROCmValidationSuite/       cloned upstream + build   [git-ignored]
│   ├── run_part_a.sh              driver: smoke → sweep → health → analysis
│   ├── run_tflops.sh, run_tflops_sweep.sh, run_rvs_health.sh
│   ├── analyze_rvs.py             → results/rvs_tflops.{md,csv}
│   ├── rerun_bandwidth_health.sh  re-check pebb/pbqt/babel on a quiet machine
│   ├── investigate_fp4_scaling.sh + analyze_fp4_scaling.py   fp4 N≥5 scaling follow-up
│   ├── run_fp4_investigation_and_update.sh                   chains the two + folds
│   │                                                          the verdict into rvs_tflops.md
│   └── update_rvs_summary_with_investigation.py
│
├── rccl-tests/                 Part B — RCCL collectives (RCCL only; no Megatron here)
│   ├── src/                       cloned upstream + build   [git-ignored]
│   ├── run_part_b.sh              driver: smoke → all-collective → config sweep → analysis
│   ├── run-rccl-all.sh, run-rccl-configs.sh, run-rccl-sendrecv.sh
│   └── analyze_rccl.py, plot_rccl_busbw.py   → results/rccl.{md,csv}, rccl_busbw.png
│
├── primus/                     Part C — GEMM/attention/RCCL microbenches + Megatron-LM
│   ├── Primus/                     cloned upstream          [git-ignored]
│   ├── run_part_c.sh               driver: gate → sweep → Megatron → report
│   ├── run_gpu_scan.sh, run_full_sweep.sh, run_megatron.sh
│   ├── generate_report.py          → results/PRIMUS_REPORT.md
│   └── sweep_out_<RUN_ID>/         bench output the container writes out
│
├── atom/                       Part D — ATOM LLM inference serving (3 model tiers)
│   ├── ATOM/                       cloned upstream          [git-ignored]
│   ├── README.md                   Part D docs — tiers, why TP, safety design
│   ├── run_part_d.sh               driver: gate → tier1 → tier2 → tier3 → analysis
│   ├── download_models.sh          resumable weight fetch → /mnt/scratch/shaohao/models/
│   ├── run_atom_server.sh, stop_atom_server.sh, run_atom_bench.sh
│   ├── analyze_atom.py             → results/atom.{md,csv}
│   └── run_kimi_ep_ab.sh + analyze_kimi_ep.py   expert-parallelism A/B experiment
│
├── logs/{rvs,rccl,primus,atom}/   per-run driver logs, STATE.txt, raw benchmark output
└── results/                    final deliverables
    ├── rvs_tflops.{md,csv}         Part A
    ├── fp4_investigation.md        Part A follow-up
    ├── rccl.{md,csv}, rccl_busbw.png   Part B
    ├── PRIMUS_REPORT.md            Part C
    ├── atom.{md,csv}               Part D — 3 model tiers at a glance
    └── Kimi-K3 deep-dive (Part D tier 3):
        ├── kimi-k3-improve.md          ← START HERE: all 3 runs consolidated + next steps
        ├── kimi-k3-base.md             run A — original recipe, max-num-seqs 64
        ├── kimi-k3-mad.{md,csv}        run B — AMD MAD recipe, max-num-seqs 64
        ├── kimi-k3-maxseqs.{md,csv}    run C — max-num-seqs 256 (best: 2,482 tok/s)
        └── kimi-k3-comparison.md       run A vs run B detail
```

Megatron-LM is benchmarked **only** through Primus (Part C). Part B mirrors
[`../dell-cloud/rccl-tests/`](../dell-cloud/rccl-tests/); the Megatron-LM training sweep in
`dell-cloud/megatron-lm/` is not reproduced. Part D has no dell-cloud counterpart — it's
net-new LLM-inference characterization, not a reproduction.

Each part's `run_part_*.sh` is a self-contained driver: idle-GPU guard, every stage in
order, analysis at the end, safe to run under `nohup`/`setsid` unattended. Cloned upstream
sources (ROCmValidationSuite, rccl-tests, Primus, ATOM) sit inside their part directories but
are git-ignored — see [`.gitignore`](.gitignore). Bulk regenerable caches (Docker layers,
Triton/HF/pip JIT caches, the analysis venv, **model weights**) live at
`/mnt/scratch/shaohao/`, off-repo — see [`notes.md`](notes.md) for exactly what's where.

## Where the suites overlap

Two workloads are measured by more than one suite. That is deliberate: the suites sit at
different layers of the stack, so the *gap* between them is itself a result. Neither number
is "more correct" — they answer different questions.

### GEMM — RVS `gst` vs Primus `gemm`

| | RVS `gst` (Part A) | Primus `gemm` family (Part C) |
|---|---|---|
| Question it answers | Is the silicon healthy, and what is the ceiling? | What does a training framework actually achieve? |
| Stack | hipBLASLt called from C++ — no framework | PyTorch → hipBLASLt, launched under `torchrun` |
| Shapes | Fixed large (8192×8192×16384), `rotating: 512` buffers to defeat L2 reuse | Training-shaped; `gemm-dense` / `gemm-deepseek` use real model layer shapes |
| Precisions | 9 — `fp4 fp6 bf6 fp8 bf8 fp16 bf16 fp32 fp64` | Mainly bf16 / fp8 |
| Metric | Per-GPU peak GFLOPS per log interval + a `target_stress` pass/fail | TFLOPS table |
| Multi-GPU | `parallel: true` — N *independent* stress instances, no communication | N ranks via torchrun; the GEMM itself still does no inter-GPU communication |

Expect **Primus `gemm` < RVS `gst`** at the same precision — the difference is framework and
dispatch overhead. A small gap is normal; a large one is a software finding, not a hardware
one. RVS is also the only place fp4/fp6/bf6 get exercised at all.

### Collectives — rccl-tests vs Primus `rccl`

`rccl-tests` is AMD's ROCm port of `nccl-tests` — same benchmark lineage and the same
`algbw`/`busbw` methodology.

| | rccl-tests (Part B) | Primus `benchmark rccl` (Part C) |
|---|---|---|
| Layer | Raw RCCL C API | `torch.distributed`, RCCL underneath |
| Metric | `algbw` + `busbw` (algorithm-normalized) | `eff_gbps` |
| Message sizes | Sweeps 16 MiB → 8 GiB, ×2 steps | Fixed training-relevant set |
| Correctness | Verified (`-c 1`) | Not the focus |
| Process model | **1 process driving N GPUs** (`-g N`, `MPI=0`) | **N processes, 1 GPU each** (torchrun) |

The process-model row is the one that bites. Single-process-multi-device and
one-rank-per-process take **different code paths inside RCCL** — different communicator
setup and stream usage. Real training is the N-process model. So if Part B shows a clean
curve at some N but Primus `rccl` cliffs at the same N, suspect the process model before the
fabric.

`busbw` is the more comparable number across N, because it normalizes out the fact that
different collectives move different amounts of data for the same buffer size — which is why
the non-power-of-2 cliff analysis in `analyze_rccl.py` keys on it rather than on `algbw`.

### Reading a disagreement

The suites form a ladder, and each rung licenses a conclusion about the one above it:

1. **RVS health** (`pbqt` XGMI, `pebb` PCIe, `babel` HBM, `mem`) — is the fabric and memory
   sound? A clean result here is what lets a later RCCL cliff be attributed to the
   *algorithm layer* rather than to hardware.
2. **RVS `gst`** — the compute ceiling per precision.
3. **rccl-tests** — the collective ceiling at the RCCL API.
4. **Primus microbenches** — what PyTorch gets from 2 and 3.
5. **Megatron llama2-7B** — what a real model gets from all of the above.

A drop between adjacent rungs localizes the problem to that rung. This is the whole reason
Part A runs first and Part C runs last.

### No overlap

Each suite also owns something the others cannot see: RVS has the low-precision
(fp4/fp6/bf6) ceiling and the hardware health verdict; rccl-tests has the full message-size
curve and correctness verification; Primus has attention, the DeepSeek/dense GEMM shapes, and
the only end-to-end training number.

> Caveat: the RVS rows above were read from the generated `gst` conf in `run_tflops.sh`, and
> the rccl-tests rows from its CLI. The Primus rows come from the CLI surface and
> `dell-cloud/primus/REPORT.md` — Primus' `gemm`/`rccl` implementation source has not been
> read, so treat its exact shapes and the precise definition of `eff_gbps` as unconfirmed
> until the first run lands.

Primus also ships a **JAX/MaxText** path (`requirements-jax.txt`,
`examples/maxtext/configs/MI355X/`). It is **not run here** — Part C reproduces the Megatron
and microbench portions of `dell-cloud/primus/REPORT.md` only. Likewise unused:
`strided-allgather` (new in v26.5) and the FP8 Megatron config; this is a BF16 reproduction.

## Status

| Part | State |
|------|-------|
| A — ROCm Validation Suite | ✅ Complete — `gst` sweep, health modules, fp4 N≥5 follow-up |
| B — RCCL collectives | ✅ Complete — full sweep + config sweep |
| C — Primus + Megatron-LM | ✅ Complete — microbench sweep + Megatron llama2-7B N=1..8 |
| D — ATOM (LLM inference) | ✅ Complete — 3 tiers (Qwen3-8B, Llama-70B, Kimi-K3); EP experiment run (unsupported on this build) |

All four parts ran **strictly sequentially** — each wants all 8 GPUs and the full
~11.2 kW tray power envelope, so overlapping them would invalidate every number.
