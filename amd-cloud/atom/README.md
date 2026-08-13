# Part D — ATOM (LLM inference serving)

[ATOM](https://github.com/ROCm/ATOM) is AMD's AITER-optimized, vLLM-like inference engine.
This part measures **serving throughput and latency** on 8 x MI355X — a different question
from Parts A–C, which measure raw compute (RVS), fabric bandwidth (rccl-tests), and training
throughput (Primus).

Nothing here has been run yet. Setup is complete and the scripts are staged.

## Why this part is shaped differently

| | Parts A–C | Part D (ATOM) |
|---|---|---|
| Measures | FLOPS / GB/s / training TF/s | req/s, tokens/s, TTFT, TPOT |
| Needs model weights | No (mock data) | **Yes** — no mock-data path exists for a server |
| Needs network at run time | No (`HF_HUB_OFFLINE=1`) | Weights fetched ahead of time; run is offline |
| `dell-cloud/` baseline | Yes — this repo reproduces it | **None** — net-new characterization |
| Comparison point | dell-cloud numbers | [ATOM public dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/) |

Because there is no baseline in this repo, Part D's numbers are **absolute, not comparative**.
Do not put them in the same table as the Part A–C results.

## Status

| Item | State |
|---|---|
| `ATOM/` upstream clone | ✅ HEAD `6a756fdb` (git-ignored) |
| `rocm/atom-dev:latest` image | ✅ **already on host** (106 GB) — no pull needed |
| Tier-1 model `Qwen3-8B-FP8` | ✅ 8.9 GB → `/mnt/scratch/shaohao/models/Qwen3-8B-FP8` |
| Tier-2 model `Llama-3.1-70B-Instruct-FP8` | ✅ 68 GB, 15 shards → `/mnt/scratch/shaohao/models/Llama-3.1-70B-Instruct-FP8` |
| Scripts | ✅ 5 scripts, syntax-checked; analyzer tested on synthetic JSON |
| gfx950 gate | ⏸ not run (touches GPUs) |
| Benchmark run | ⏸ **awaiting go** |

The image was already present before this work started — it is the 106 GB image the plan's
disk warning flagged as "not ours". That means **the box is shared**, which drives the safety
design below.

## Scripts

| Script | Purpose |
|---|---|
| `run_atom_server.sh <model> [TP] [PORT]` | Start the server in a container; waits for `/v1/models` **and** actual VRAM allocation |
| `stop_atom_server.sh` | Stop/remove our container by name |
| `run_atom_bench.sh <model> [PORT] [ISL] [OSL] [CONC]` | Concurrency sweep against a running server |
| `analyze_atom.py <sweep_dir>... -o results` | → `results/atom.{md,csv}` + knee detection |

### These are not wrappers around ATOM's own scripts, deliberately

ATOM ships `scripts/start_atom_server.sh` and `scripts/run_benchmark.sh`. They are not safe
to run unmodified here:

1. **`start_atom_server.sh` opens with `pkill -f 'atom.entrypoints'` and
   `pkill -9 -f 'multiprocessing.spawn'`.** On a shared box that kills *any* user's ATOM
   server and any unrelated Python multiprocessing job. `run_atom_server.sh` **refuses to
   start** if it finds a foreign ATOM process, a busy GPU, or a taken port — it never kills
   anything it did not create, and `stop_atom_server.sh` only ever acts on our container by
   name.
2. **`KINETO_CONFIG=/home/ljin1/dk/libkineto.conf`** is hardcoded to another user's home
   directory.
3. **`OUTPUT_DIR=/app/logs_claude`** is hardcoded, and `run_benchmark.sh` overwrites a single
   `benchmark.log` per invocation. Our sweep writes one `c<N>.json` + `c<N>.log` per
   concurrency into the tracked log tree.

We call `atom.benchmarks.benchmark_serving` directly, which is what their script does
underneath — so the measurement path is theirs, only the plumbing is ours.

## Running it (when given the go)

**Unattended (recommended)** — gate → tier 1 → tier 2 → analysis, strictly sequential:

```bash
cd /home/amd/shaohao/amd-benchmarks/amd-cloud/atom
nohup ./run_part_d.sh all > $LOG_ROOT/atom/part_d.out 2>&1 &
```

`run_part_d.sh` aborts up front if any GPU is busy or a foreign ATOM server is running, runs
the gfx950 gate before committing to anything, brings up exactly one server at a time, and
stops each before the next. Progress lands in `logs/atom/part_d_*/STATE.txt`.

**Manual, step by step:**

```bash
cd /home/amd/shaohao/amd-benchmarks/amd-cloud && source common/env.sh
cd atom

# 0. gate: confirm the image is gfx950-native and sees 8 GPUs (needs idle GPUs)
docker run --rm $(dgpu_args) rocm/atom-dev:latest \
  python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.device_count())"

# 1. tier-1 smoke: 8B on a single GPU
./run_atom_server.sh $SCRATCH/models/Qwen3-8B-FP8 1 8000
./run_atom_bench.sh  $SCRATCH/models/Qwen3-8B-FP8 8000 1024 1024 "1 4 16 64"
./stop_atom_server.sh

# 2. tier-2 headline: 70B across all 8 GPUs
./run_atom_server.sh $SCRATCH/models/Llama-3.1-70B-Instruct-FP8 8 8001
./run_atom_bench.sh  $SCRATCH/models/Llama-3.1-70B-Instruct-FP8 8001 1024 1024 "1 2 4 8 16 32 64 128 256"
./stop_atom_server.sh

# 3. analysis
$PY analyze_atom.py $LOG_ROOT/atom/sweep_* -o $BENCH_ROOT/results
```

Full sweep is roughly 30–45 min per tier. Tier 2 additionally pays a **multi-minute model
load** (68 GB off NVMe, sharded across 8 ranks) before the first request.

## Model tiers

| Tier | Model | Size | TP | Notes |
|---|---|---|---|---|
| 1 | `Qwen/Qwen3-8B-FP8` | 8.9 GB | 1 | ✅ Single-GPU smoke — proves the path, says nothing about the box |
| 2 | `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` | 68 GB | 8 | ✅ Dense headline — TP8 puts RCCL in the critical path |
| 3 | `moonshotai/Kimi-K3` | **1.56 TB** | 8 | ✅ Frontier MoE — 2.78T params, the hardest thing this box can hold |

**All three tiers will be run**, in that order, via `./run_part_d.sh all`.

Each tier answers a different question. Tier 1 proves the serving path on one GPU. Tier 2 is
the dense TP8 case: every layer all-reduces, so its latency is coupled to Part B's collective
results — a non-power-of-2 cliff there would surface here as user-visible latency. Tier 3 is
the capability statement: a 2.78T-parameter model resident on a single node, which is the
thing 288 GB/GPU of HBM exists to make possible.

### Tier 3 — Kimi-K3 specifics

`KimiK3ForConditionalGeneration`: a KimiLinear hybrid-attention MoE. Each decoder layer is
either a **KDA linear-attention** layer or an **MLA full-attention** layer, over an MXFP4
latent MoE. 1.56 TB of weights across 8 × 288 GB = 2304 GB HBM, so it fits with room for KV
cache but not much slack.

The launch flags in `run_part_d.sh` are taken verbatim from `ATOM/recipes/Kimi-K3.md` and are
**not optional tuning knobs**:

| Flag | Why |
|---|---|
| `-tp 8` | Required for the model to fit at all |
| `--gpu-memory-utilization 0.93` | So the CUDA-graph pool fits beside the KDA per-request state cache |
| `--no-enable_prefix_caching` | KDA recurrent state is per-request and **cannot** be reconstructed from the paged MLA cache — prefix reuse would be silently incorrect, not just slow |
| `--online_quant_config` (PTPC-FP8) | Quantizes attention/dense at load; routed MoE experts are already MXFP4 in the checkpoint and are excluded |
| `--max-num-seqs 64`, `--max-model-len 16384` | Recipe-validated operating point |

Because `max-num-seqs` is 64, the tier-3 concurrency sweep stops at 64 (`KIMI_CONC`). Driving
past `max-num-seqs` measures the request queue, not the engine.

Expect a **long model load** — 1.56 TB off NVMe sharded across 8 ranks — so tier 3 runs with
`READY_TIMEOUT=2400` (40 min) rather than the default 600 s.

### Why RedHatAI and not meta-llama

`meta-llama/Llama-3.1-70B-Instruct` is `gated: manual` — it needs an `HF_TOKEN` *and*
per-account license approval. `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` is **ungated**, is
the canonical vLLM-ecosystem FP8 W8A8 quantization of the same weights, and declares
`quant_method: compressed-tensors` — one of the formats ATOM's quantization dispatch
recognizes (`compressed-tensors`, `fp8`, `quark`, `mxfp4`). No token was required.
`nvidia/Llama-3.1-70B-Instruct-FP8` is also ungated but ships a TensorRT-LLM/ModelOpt-flavoured
quant, so it was not chosen.

At 68 GB across 8 GPUs that is ~9 GB of weights per GPU against 288 GB of HBM each — leaving
an unusually large KV-cache budget, so expect high achievable concurrency.

Weights live on `/mnt/scratch/shaohao/models` (6.6 TB free), **never on `/`** — which has
~161 GB free and also holds `/var/lib/docker`.

> ⚠️ The Qwen3-8B-FP8 recipe in `ATOM/recipes/Qwen3-8B-FP8.md` is written for **RX 9070 XT
> (gfx1201)** and sets `ATOM_USE_UNIFIED_ATTN=1`, `ATOM_USE_TRITON_GEMM=1` and friends to
> force Triton fallbacks on a consumer part. **Do not copy those env vars to MI355X** — they
> would bypass the tuned AITER ASM/CK kernels that are the entire point of running ATOM on
> Instinct, and would understate this hardware. Our scripts set none of them.

## Ordering constraint

Part D must not overlap Parts A–C. RVS `gst` in particular is power-bound against the
11.2 kW tray budget and reports *peak* GFLOPS — a co-resident inference server would steal
power headroom and silently depress those numbers rather than fail loudly.
`run_atom_server.sh` enforces this by refusing to start when any GPU is busy.

Downloading weights is safe at any time: it is network and disk I/O to a separate NVMe, with
no GPU involvement.
