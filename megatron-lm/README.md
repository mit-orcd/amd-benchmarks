# Megatron-LM BF16 Throughput Sweep — 1..8× MI355X

A self-contained launcher that runs `pretrain_gpt.py` from the ROCm Megatron-LM
fork inside the `megatron-lm.sif` Singularity container and reports
**TFLOP/s/GPU at N_GPUS = 1, 2, 3, 4, 5, 6, 7, 8**.

The model and parallelism are tuned for BF16 throughput on a single node of
AMD Instinct MI355X (gfx950, 288 GB HBM3e). Each sweep point uses pure data
parallelism with a distributed optimizer, no activation recomputation,
flash-attention, and xGMI peer-to-peer for collectives. The sweep is **weak
scaling**: `GBS = MICRO_BS × N_GPUS`, so per-GPU work is constant across N
and any drop in TFLOP/s/GPU as N grows is the cost of the DP collectives.

## Layout

```
/home/v89592/shaohao/megatron-lm/
├── megatron-lm.sif         # Singularity image (ROCm + PyTorch + Megatron deps)
├── Megatron-LM/            # ROCm fork, rocm_dev branch (bind-mounted into the container)
└── work/
    ├── run.sh              # sweep launcher (this script)
    ├── README.md           # you are here
    └── logs/
        └── sweep_<STAMP>/  # one directory per sweep invocation
            ├── bench_bf16_n1.log .. bench_bf16_n8.log
            └── sweep_summary.txt
```

## Requirements

- `singularity` (or `apptainer`) on the host — the script uses `singularity exec --rocm`.
- `megatron-lm.sif` present at `/home/v89592/shaohao/megatron-lm/megatron-lm.sif`.
- `Megatron-LM/` checked out at `/home/v89592/shaohao/megatron-lm/Megatron-LM` (ROCm fork).
- 8× MI355X visible on the host (`rocm-smi` should list them).

No sudo is needed. The script does not modify host state outside `work/logs/`.

## Usage

Foreground (8 runs back-to-back, ~5 min each → ~40 min total):

```bash
bash /home/v89592/shaohao/megatron-lm/work/run.sh
```

Detached (recommended — survives SSH disconnect, all output goes to `log.run`):

```bash
cd /home/v89592/shaohao/megatron-lm/work
nohup bash run.sh > log.run 2>&1 &
tail -f log.run        # follow progress; Ctrl-C just stops tailing
```

That's it. The script:

1. Validates the `.sif` and source tree exist.
2. For each `N_GPUS` in `1 2 3 4 5 6 7 8`:
   - Sets `HIP_VISIBLE_DEVICES=0,…,N_GPUS-1` and `GBS = MICRO_BS × N_GPUS`.
   - Launches `singularity exec --rocm` with the right bind mounts and
     RCCL/xGMI env variables.
   - Runs `torchrun --standalone --nproc_per_node=$N_GPUS` against
     `pretrain_gpt.py` for `TRAIN_ITERS=50` iterations (5 LR-warmup steps).
   - Tees that run's output to `logs/sweep_<STAMP>/bench_bf16_n<N>.log`.
   - Pulls `last` / `best` TFLOP/s/GPU from the log and appends one row to
     `logs/sweep_<STAMP>/sweep_summary.txt`.
3. At the end, prints the sweep summary:
   ```
   N_GPUS  GBS    last_TF/GPU   best_TF/GPU   log
   1       2      ...           ...           .../bench_bf16_n1.log
   2       4      ...           ...           .../bench_bf16_n2.log
   ...
   8       16     239.8         239.8         .../bench_bf16_n8.log
   ```

Each per-iteration TFLOP/s number is logged by Megatron itself
(`--log-throughput`) and is grepped out of each per-N log at the end.
A failure at one N is reported and the sweep continues with the next N.

## What's configured

| Knob | Value | Rationale |
| --- | --- | --- |
| GPUs | `GPU_COUNTS=(1 2 3 4 5 6 7 8)` (single node) | rendezvous via `torchrun --standalone` |
| Parallelism | TP=1, PP=1, DP=N_GPUS | pure DP — collectives stay on xGMI |
| Optimizer | `--use-distributed-optimizer` | shard Adam state across DP ranks |
| Precision | BF16 | matches MI355X peak path, no TE/FP8 needed |
| Model | L=40, H=6144, FFN=16384, heads=48 (GQA-8) | large GEMMs, fits in HBM w/o recompute |
| Seq length | 4096 | enough attention work to be compute-bound |
| Micro-batch | 2 (GBS = 2 × N_GPUS, weak scaling) | one micro-batch per GPU, no grad accum |
| Attention | flash-attn | `--use-flash-attn` |
| Recompute | off | `--log-throughput` does not credit recompute FLOPS |
| Interconnect | xGMI + SHM | `NCCL_IB_DISABLE=1`, `RCCL_MSCCL_ENABLE=1` |

## Megatron flags — what and why

These are the flags the script passes to `pretrain_gpt.py`, grouped by
purpose. The "why" is always *for a throughput benchmark on a single MI355X
node* — the same flags would not be appropriate for a real training run.

### Model architecture (`MODEL_ARGS`)

Defines a ~16 B-parameter GPT in the Llama-3 family (RoPE + SwiGLU + RMSNorm
+ GQA). Shape picked so dense GEMMs are big enough to be matmul-bound on
MI355X while the whole model still fits in 288 GB HBM without recompute.

| Flag | Value | Why |
| --- | --- | --- |
| `--num-layers` | 40 | Depth that keeps the optimizer state plus activations under HBM budget. |
| `--hidden-size` | 6144 | Large enough that QKV/MLP GEMMs hit the matrix engine, not the memory pipe. |
| `--ffn-hidden-size` | 16384 | SwiGLU FFN width (~2.67× hidden) — dominates FLOPs/layer. |
| `--num-attention-heads` | 48 | head_dim = 6144 / 48 = 128, the FlashAttention sweet spot. |
| `--group-query-attention` + `--num-query-groups 8` | GQA-8 | 8 KV groups → 6× fewer KV projections and a 6× smaller KV cache, no quality cost at this scale. |
| `--seq-length` | 4096 | Long enough that attention is non-trivial work; short enough that the activation footprint at MICRO_BS=2 stays in HBM. |
| `--max-position-embeddings` | 4096 | Matches seq-length (required for RoPE). |
| `--position-embedding-type rope` | rope | Modern default; no learned-PE memory. |
| `--swiglu` | on | SwiGLU FFN — what current frontier models use; the FLOP formula behind `--log-throughput` assumes it. |
| `--normalization RMSNorm` | RMSNorm | Cheaper than LayerNorm, matches Llama-style models. |
| `--untie-embeddings-and-output-weights` | on | Decouples input embedding from LM head — also eliminates the cross-replica embedding all-reduce (the iter-45 timer for `embedding-grads-all-reduce` is ~0.01 ms because of this). |

### Data + tokenizer (`TRAIN_ARGS`, data block)

We don't want to measure dataset I/O — only compute + comms.

| Flag | Value | Why |
| --- | --- | --- |
| `--mock-data` | on | Generates random token IDs on-GPU — zero disk / network in the hot path. |
| `--tokenizer-type NullTokenizer` | NullTokenizer | No-op tokenizer; pairs with `--mock-data`. |
| `--vocab-size 50304` | 50304 | Padded to a multiple of 128 so the LM-head GEMM is hardware-friendly. |

### Parallelism + sharding

| Flag | Value | Why |
| --- | --- | --- |
| `--tensor-model-parallel-size 1` | 1 | TP would add per-layer all-reduces on every forward and backward — we explicitly keep TP off so cross-GPU traffic is only the DP step boundary, isolating xGMI overhead. |
| `--pipeline-model-parallel-size 1` | 1 | PP would add bubble + send/recv; for a single-node benchmark it only hurts throughput. |
| `--data-parallel-sharding-strategy no_shard` | no_shard | Full replication of params + grads across DP ranks. Combined with `--use-distributed-optimizer`, only Adam state is sharded. |
| `--use-distributed-optimizer` | on | Shards the fp32 Adam state across DP ranks (ZeRO-1). Makes the 16 B model fit comfortably and adds exactly one `params-all-gather` per step — the ~79 ms timer in the steady-state breakdown. |

### Batching + schedule

A 50-step run with 5 LR-warmup steps is enough to see steady-state TFLOP/s.

| Flag | Value | Why |
| --- | --- | --- |
| `--micro-batch-size 2` | 2 | One micro-batch per GPU; no gradient accumulation, so the timing trace cleanly maps to one forward + one backward + one optimizer step. |
| `--global-batch-size` | `MICRO_BS × N_GPUS` | Weak scaling: per-GPU work is constant across the sweep. |
| `--train-iters 50` | 50 | Long enough that warmup is amortized; short enough to keep wall time per N around 3 min. |
| `--lr 3e-4` / `--min-lr 3e-5` | — | Reasonable defaults; the loss curve isn't the metric here. |
| `--lr-decay-style cosine` | cosine | Required when `--lr-decay-iters` is set. |
| `--lr-warmup-iters 5` | 5 | Excludes the first few iters from the steady-state TFLOP/s. |
| `--lr-decay-iters 30` | 30 | Decay completes before iter 50 so the final samples are steady-state at `min-lr`. |
| `--weight-decay 0.1` / `--adam-beta1 0.9` / `--adam-beta2 0.95` / `--clip-grad 1.0` | Llama-style defaults | Fine for a synthetic run. |

### Precision + kernels

| Flag | Value | Why |
| --- | --- | --- |
| `--bf16` | on | MI355X's peak BF16 path (~5 PFLOPS/GPU); no loss-scaling needed unlike FP16. |
| `--use-flash-attn` | on | FlashAttention 2/3 instead of vanilla attention — the only way attention stays compute-bound at seq=4096. |

### Eval / save / logging

These are the "make-the-benchmark-not-do-other-stuff" flags.

| Flag | Value | Why |
| --- | --- | --- |
| `--eval-interval 1000000` | huge | Effectively disables validation eval inside the 50-iter window. |
| `--save-interval 1000000` | huge | Disables checkpointing — no disk writes. |
| `--log-interval 5` | 5 | Print one throughput line every 5 iters → 10 steady-state samples. |
| `--log-throughput` | on | Causes the `throughput per GPU (TFLOP/s/GPU): X` lines that the script greps for the summary. |
| `--timing-log-level 2` | 2 | Emit per-stage timers (`forward-compute`, `backward-compute`, `all-grads-sync`, `params-all-gather`, …). This is how the comm overhead is measured. |
| `--timing-log-option all` | all | Report timers per rank, not just the average — lets us check rank balance / stragglers. |

## Tuning

Open `run.sh` and edit the variables near the top — everything else
follows from them.

- **OOM** (CUDA/HIP out of memory): drop `MICRO_BS` from 2 → 1 first; if still
  tight, shrink `HIDDEN`/`FFN`/`NUM_LAYERS`, or as a last resort add
  `--recompute-activations --recompute-granularity selective` to `TRAIN_ARGS`
  (this will *lower* the reported TFLOPS number).
- **Want higher reported TFLOPS**: try `SEQ_LEN=8192` (more attention work),
  or bump `HIDDEN` to 8192 / `FFN` to 28672 with `NUM_LAYERS=32`. Watch HBM.
- **Different GPU sweep**: edit `GPU_COUNTS=(...)` in `run.sh` (e.g.
  `GPU_COUNTS=(8)` to reproduce the old single-point run, or
  `GPU_COUNTS=(1 2 4 8)` for power-of-two only). `GBS` recomputes per N.
- **Strong scaling instead of weak**: pin `GBS` to a constant inside the loop
  (e.g. `GBS=16`) instead of `GBS=$(( MICRO_BS * N_GPUS ))`. Megatron requires
  `GBS % (N_GPUS * MICRO_BS) == 0`, so adjust `MICRO_BS` accordingly.
- **Multi-node**: replace `--standalone` with `--rdzv_backend=c10d
  --rdzv_endpoint=<master>:<port>` and bump `N_NODES`. You'll also want to
  re-enable IB (`NCCL_IB_DISABLE=0`, set `NCCL_IB_HCA`) and pick the right
  `NCCL_SOCKET_IFNAME`.

## Log files

Each sweep invocation creates `logs/sweep_<STAMP>/` containing:

- `bench_bf16_n<N>.log` — one file per N_GPUS, each with:
  - Container Python / PyTorch / `HIP_VISIBLE_DEVICES` banner.
  - Megatron's startup arg dump.
  - Per-`--log-interval` iteration line, including
    `throughput per GPU (TFLOP/s/GPU): <value>`.
  - Per-iteration rank-by-rank timer breakdown (`--timing-log-level 2`):
    `forward-compute`, `backward-compute`, `all-grads-sync`,
    `params-all-gather`, `optimizer`, etc. — this is where the comm
    overhead at each N is read off.
  - Optional RCCL warnings (`NCCL_DEBUG=WARN`).
- `sweep_summary.txt` — one row per N with `N_GPUS | GBS | last_TF/GPU |
  best_TF/GPU | log`. Printed to stdout at the end of the run as well.

If a single N fails (e.g. OOM, RCCL init error) the script writes `ERR` to
that row of `sweep_summary.txt`, logs the exit code, and moves on to the
next N rather than aborting the sweep.
