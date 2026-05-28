# Megatron-LM BF16 Throughput Benchmark — 8× MI355X

A self-contained launcher that runs `pretrain_gpt.py` from the ROCm Megatron-LM
fork inside the `megatron-lm.sif` Singularity container and reports
**TFLOP/s/GPU**.

The model and parallelism are tuned to maximize achieved BF16 TFLOPS on a
single node of 8× AMD Instinct MI355X (gfx950, 288 GB HBM3e). It uses pure
data parallelism with a distributed optimizer, no activation recomputation,
flash-attention, and xGMI peer-to-peer for collectives.

## Layout

```
/home/v89592/shaohao/megatron-lm/
├── megatron-lm.sif         # Singularity image (ROCm + PyTorch + Megatron deps)
├── Megatron-LM/            # ROCm fork, rocm_dev branch (bind-mounted into the container)
└── work/
    ├── run.sh              # launcher (this script)
    ├── README.md           # you are here
    └── logs/               # one log file per run, timestamped
```

## Requirements

- `singularity` (or `apptainer`) on the host — the script uses `singularity exec --rocm`.
- `megatron-lm.sif` present at `/home/v89592/shaohao/megatron-lm/megatron-lm.sif`.
- `Megatron-LM/` checked out at `/home/v89592/shaohao/megatron-lm/Megatron-LM` (ROCm fork).
- 8× MI355X visible on the host (`rocm-smi` should list them).

No sudo is needed. The script does not modify host state outside `work/logs/`.

## Usage

```bash
bash /home/v89592/shaohao/megatron-lm/work/run.sh
```

That's it. The script:

1. Validates the `.sif` and source tree exist.
2. Launches `singularity exec --rocm` with the right bind mounts and RCCL/xGMI
   env variables.
3. Runs `torchrun --standalone --nproc_per_node=8` against `pretrain_gpt.py`
   for `TRAIN_ITERS=50` iterations (5 warmup steps via the LR schedule).
4. Tees all output to `work/logs/bench_bf16_<timestamp>.log`.
5. At the end, prints a throughput summary:
   ```
   ==== throughput summary ====
   samples : 10
   last    : 1234.5 TFLOP/s/GPU
   best    : 1241.2 TFLOP/s/GPU
   ```

Each per-iteration TFLOP/s number is logged by Megatron itself
(`--log-throughput`) and is grepped out of the log file at the end.

## What's configured

| Knob | Value | Rationale |
| --- | --- | --- |
| GPUs | 8 (single node) | rendezvous via `torchrun --standalone` |
| Parallelism | TP=1, PP=1, DP=8 | pure DP — collectives stay on xGMI |
| Optimizer | `--use-distributed-optimizer` | shard Adam state across DP ranks |
| Precision | BF16 | matches MI355X peak path, no TE/FP8 needed |
| Model | L=40, H=6144, FFN=16384, heads=48 (GQA-8) | large GEMMs, fits in HBM w/o recompute |
| Seq length | 4096 | enough attention work to be compute-bound |
| Micro-batch | 2 (global 16) | one micro-batch per GPU, no grad accum |
| Attention | flash-attn | `--use-flash-attn` |
| Recompute | off | `--log-throughput` does not credit recompute FLOPS |
| Interconnect | xGMI + SHM | `NCCL_IB_DISABLE=1`, `RCCL_MSCCL_ENABLE=1` |

## Tuning

Open `run.sh` and edit the variables near the top — everything else
follows from them.

- **OOM** (CUDA/HIP out of memory): drop `MICRO_BS` from 2 → 1 first; if still
  tight, shrink `HIDDEN`/`FFN`/`NUM_LAYERS`, or as a last resort add
  `--recompute-activations --recompute-granularity selective` to `TRAIN_ARGS`
  (this will *lower* the reported TFLOPS number).
- **Want higher reported TFLOPS**: try `SEQ_LEN=8192` (more attention work),
  or bump `HIDDEN` to 8192 / `FFN` to 28672 with `NUM_LAYERS=32`. Watch HBM.
- **Different GPU count**: change `N_GPUS`; `GBS` recomputes from it.
- **Multi-node**: replace `--standalone` with `--rdzv_backend=c10d
  --rdzv_endpoint=<master>:<port>` and bump `N_NODES`. You'll also want to
  re-enable IB (`NCCL_IB_DISABLE=0`, set `NCCL_IB_HCA`) and pick the right
  `NCCL_SOCKET_IFNAME`.

## Log file

The log contains:

- Container Python / PyTorch / device-count banner.
- Megatron's startup arg dump.
- Per-`--log-interval` iteration line, including
  `throughput per GPU (TFLOP/s/GPU): <value>`.
- Optional RCCL warnings (`NCCL_DEBUG=WARN`).

If a run fails, the log path is printed before the script exits with the
torchrun return code.
