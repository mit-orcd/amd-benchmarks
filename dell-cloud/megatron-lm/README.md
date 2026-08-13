# Megatron-LM MI355X Benchmark — v26.1

Scripts and instructions to reproduce `summary.md` on a single node of 8× AMD
Instinct MI355X (gfx950, 288 GB HBM3e). Run on the Dell Cloud server — see
[`../README.md`](../README.md).

> The RCCL collective sweeps that used to live here have moved to
> [`../rccl-tests/`](../rccl-tests/). They ran from this same working directory and
> container, so their documents still reference `work/` paths; the split is by subject —
> RCCL collectives there, Megatron-LM training here. The N=5/6/7 cliff analysis
> (`summary-power2.md`, `summary-rccl.md`, `notes-amd.md`) is in that directory.

`summary.md` is produced by three sweeps run in order:

| summary.md section | Script |
|--------------------|--------|
| §1 Headline / GPU-count curve | `run-tflops-v26.1-gpusweep.sh` |
| §2 MBS × Recompute × Precision | `run-tflops-v26.1.sh` |
| §3 N=8 tuning ablations | `run-tflops-v26.1-tune.sh` (then `run-tflops-v26.1-tune-resume.sh` if interrupted) |

## Layout

```
/home/v89592/shaohao/megatron-lm/
├── megatron-lm-v26.1.sif           # Singularity image (~21 GB)
├── Megatron-LM/                    # ROCm fork, rocm_dev branch (bind-mounted)
└── work/
    ├── run-tflops-v26.1-gpusweep.sh
    ├── run-tflops-v26.1.sh
    ├── run-tflops-v26.1-tune.sh
    ├── run-tflops-v26.1-tune-resume.sh
    ├── summary.md
    └── logs/
        ├── tflops_v26.1_gpusweep_<STAMP>/   # §1 outputs
        ├── tflops_v26.1_<STAMP>/            # §2 outputs
        └── tflops_v26.1_tune_<STAMP>/       # §3 outputs
```

## Requirements

- `singularity` (or `apptainer`) on the host — scripts use `singularity exec --rocm`.
- ROCm 7.2.3 host driver; `rocm-smi` must see all 8 GPUs.
- No sudo needed. Scripts write only under `work/logs/`.

## Installation

### 1. Clone Megatron-LM (ROCm fork)

```bash
cd /home/v89592/shaohao/megatron-lm
git clone -b rocm_dev https://github.com/ROCm/Megatron-LM.git Megatron-LM
```

The measured commit is `705c37b83`; HEAD of `rocm_dev` tracks close to it.
Pin to the exact commit for strict reproducibility:

```bash
cd Megatron-LM && git checkout 705c37b83
```

### 2. Build the Singularity image

```bash
cd /home/v89592/shaohao/megatron-lm
singularity build megatron-lm-v26.1.sif docker://rocm/megatron-lm:v26.1
```

This pulls the `rocm/megatron-lm:v26.1` Docker image (~21 GB) and converts it
to a SIF. The build takes ~15 min and is a one-time step. The image contains
gfx950 native fat-binaries (hipBLASLt, aiter assembly), PyTorch 2.10.0.dev,
and TransformerEngine 2.6; `HSA_OVERRIDE_GFX_VERSION` is not needed.

## Running the sweeps

All three scripts are self-contained. Run them detached so they survive SSH
disconnect — each sweep takes 40–90 min.

### §1 — GPU-count sweep (N=1..8, BF16, MBS=4)

Produces the weak-scaling table in `summary.md §1` (TF/s/GPU at each N).
Uses an OOM-fallback chain at low N (full recompute if no-RC OOMs).

```bash
cd /home/v89592/shaohao/megatron-lm/work
nohup bash run-tflops-v26.1-gpusweep.sh > log.tflops-v26.1-gpusweep 2>&1 &
tail -f log.tflops-v26.1-gpusweep
```

Per-N logs land in `logs/tflops_v26.1_gpusweep_<STAMP>/`; the summary table
is at `tflops_summary.txt` in that directory and printed at the end.

### §2 — MBS × Recompute × Precision sweep (N=8)

Sweeps `(MBS, recompute, precision)` at fixed N=8 to find the per-config
peak. Produces the table in `summary.md §2` (BF16 and FP8 configs including
the 1,108 TF/s/GPU FP8 winner).

```bash
cd /home/v89592/shaohao/megatron-lm/work
nohup bash run-tflops-v26.1.sh > log.tflops-v26.1 2>&1 &
tail -f log.tflops-v26.1
```

Per-config logs: `logs/tflops_v26.1_<STAMP>/bench_mbs{N}_rc{mode}_{prec}.log`.
Summary: `tflops_summary.txt` in that directory.

### §3 — N=8 tuning ablations

Ablates one knob at a time against the §2 baseline (DDP bucket size, NCCL
buffer, hipBLASLt tuning depth, FP8 recipe). Produces `summary.md §3`.

Start the full sweep:

```bash
cd /home/v89592/shaohao/megatron-lm/work
nohup bash run-tflops-v26.1-tune.sh > log.tflops-v26.1-tune 2>&1 &
tail -f log.tflops-v26.1-tune
```

If the host crashes mid-sweep (MSCCL knob is a known crash trigger — see
`summary.md §3`), resume with the companion script, which skips MSCCL and
appends results into the same `logs/tflops_v26.1_tune_<STAMP>/` directory:

```bash
nohup bash run-tflops-v26.1-tune-resume.sh > log.tflops-v26.1-tune-resume 2>&1 &
tail -f log.tflops-v26.1-tune-resume
```

**Do not enable `RCCL_MSCCL_ENABLE=1`** on this fabric — it has crashed the
host twice (N=7 in the GPU-count sweep, and during `bf16_msccl_on` in the
tuning sweep), each causing ~3 h downtime.

## Tuning caches

The scripts create and reuse `~/.tune-v26.1/hipblaslt/` and
`~/.tune-v26.1/miopen/` as persistent tuning-cache directories. These are
bind-mounted into the container via `--bind $ROOT:$ROOT` so warm-path kernel
selections carry over between sweep invocations. No manual setup is needed;
the directories are created automatically on first run.
