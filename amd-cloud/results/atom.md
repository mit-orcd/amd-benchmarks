# ATOM serving benchmark — MI355X

Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on 8 x MI355X (gfx950), ROCm 7.14. Workload: ISL/OSL 1024/1024, `--ignore-eos`, saturating request rate.

Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric bandwidth. There is no `dell-cloud/` baseline for this suite — compare against ATOM's public dashboard, not against this repo.

## Summary — three models at a glance

| Tier | Model | Params (total / active) | On disk | TP | Peak tok/s | @ conc | TTFT med @c=1 (ms) | TPOT med @c=1 (ms) | Knee |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tier 1 | `Qwen3-8B-FP8` | 8 B / 8 B | 8.9 GB | 1 | **14,962.7** | 256 | 45.8 | 6.24 | none in range |
| tier 2 | `Llama-3.1-70B-Instruct-FP8` | 70 B / 70 B | 68 GB | 8 | **9,342.4** | 256 | 71.9 | 9.81 | none in range |
| tier 3 | `Kimi-K3` | 2.78 T / ~84 B | 1.5 TB | 8 | **1,258.5** | 64 | 224.9 | 21.48 | none in range |

*Active* params are what actually fire per token — identical to total for a dense model, but only ~3% of total for Kimi-K3's MoE. That distinction drives most of the throughput differences below.

## The three models

### Tier 1 — `Qwen3-8B-FP8`

[`Qwen/Qwen3-8B-FP8`](https://huggingface.co/Qwen/Qwen3-8B-FP8) · `Qwen3ForCausalLM` · **dense**

| | |
|---|---|
| Parameters | **8 B** total, 8 B active per token |
| Checkpoint on disk | 8.9 GB |
| Quantization | FP8 (block 128) |
| Layers / hidden | 36 / 4096 |
| Attn heads / KV heads | 32 / 8 |
| Vocab / max context | 151,936 / 40K |
| Tensor parallel | TP=1 |

Smallest tier and the only single-GPU run. Ungated on HF, so it proves the serving path without a token. GQA 32:8.

### Tier 2 — `Llama-3.1-70B-Instruct-FP8`

[`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8) · `LlamaForCausalLM` · **dense**

| | |
|---|---|
| Parameters | **70 B** total, 70 B active per token |
| Checkpoint on disk | 68 GB |
| Quantization | FP8 W8A8 (compressed-tensors) |
| Layers / hidden | 80 / 8192 |
| Attn heads / KV heads | 64 / 8 |
| Vocab / max context | 128,256 / 131K |
| Tensor parallel | TP=8 |

The dense headline. At TP=8 every layer all-reduces, so this is the tier that puts RCCL in the per-token critical path. RedHatAI quant chosen because meta-llama is gated.

### Tier 3 — `Kimi-K3`

[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) · `KimiK3ForConditionalGeneration` · **MoE (hybrid attn)**

| | |
|---|---|
| Parameters | **2.78 T** total, ~84 B active per token |
| Checkpoint on disk | 1.5 TB |
| Quantization | MXFP4 experts + PTPC-FP8 rest |
| Layers / hidden | 93 / 7168 |
| Attn heads / KV heads | 96 / 96 |
| Vocab / max context | 163,840 / 1M |
| Tensor parallel | TP=8 |

Frontier MoE: 896 routed experts, top-16 + 2 shared, so only ~3% of the model fires per token. 24 MLA full-attention layers + 69 KDA linear-attention layers — only the 24 keep a growing KV cache.

### Are these numbers what the hardware should give?

**Roofline tok/s** = the most tokens/s possible if reading weights from HBM were the *only* cost. One decode step emits `batch` tokens and must read every weight it activates once, so:

```
step_time >= weight_bytes / (HBM_BW_per_GPU x GPUs)
roofline  =  batch / step_time
          =  batch x (HBM_BW_per_GPU x GPUs) / weight_bytes
```

e.g. Llama-70B: 256 x (8000 GB/s x 8) / 68 GB = **240,941 tok/s**.

`weight_bytes` is what is *actually read*, not model size — identical for dense models, but for Kimi-K3 it is **931 GB**, not 1.5 TB and not the ~84 B active params, because at batch 64 roughly 610 of 896 experts fire per layer. It deliberately ignores KV reads, compute, collectives and prefill, so it is a *loose upper bound*: far below it means weight traffic is not the constraint; near it means it is.

| Model | GPUs | Peak tok/s | tok/s **per GPU** | step (ms) | weights read/step | roofline tok/s | % of roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-8B-FP8` | 1 | 14,962.7 | **14,962.7** | 17.1 | 8.0 GB | 256,000.0 | **5.8%** |
| `Llama-3.1-70B-Instruct-FP8` | 8 | 9,342.4 | **1,167.8** | 27.4 | 68.0 GB | 240,941.2 | **3.9%** |
| `Kimi-K3` | 8 | 1,258.5 | **157.3** | 50.9 | 931.0 GB | 4,399.6 | **28.6%** |

**Yes — and the % column is the interesting part.** Kimi-K3 sits at ~29% of its weight-bandwidth ceiling while the two dense models sit at 4-6%. That is not Kimi doing better; it means Kimi is genuinely **bandwidth-bound** while Qwen and Llama are not. It also matches, independently, the ~29% HBM utilization measured in `kimi-k3.md` §3 — two different routes to the same number.

To be precise about *which* bandwidth: this is **intra-GPU HBM** — each GPU reading weights out of its own 8 TB/s on-package memory. It is **not** the XGMI GPU-to-GPU interconnect, which in the same run carries only activation all-reduces and sits at ~1% utilized. The two are often conflated; here they differ by roughly 390:1 in traffic. **See [`kimi-k3.md`](kimi-k3.md) for the full breakdown** — §3 ranks the three candidate bottlenecks (compute 1.1%, HBM ~29%, XGMI ~1.1%), §4 gives the per-step byte volumes on each path, and the terminology section at the top defines HBM vs XGMI.

#### If the dense models are not bandwidth-bound, what limits them?

**Not compute either.** Accounting for a Llama-70B decode step at c=256 (measured step time 27.4 ms):

| Candidate cost | Estimated per step | Share of step |
|---|---:|---:|
| Weight reads (68 GB / 8 GPUs at 8 TB/s) | ~1.1 ms | ~4% |
| KV-cache reads (~7.9 GB/GPU) | ~1.0 ms | ~4% |
| Decode compute (2 x 70e9 x 256 FLOP) | ~3.4 ms | ~12% |
| TP all-reduce (160 calls/step) | ~2-3 ms | ~10% |
| **Unaccounted** | **~19 ms** | **~70%** |

No single hardware resource is saturated: bandwidth ~4%, compute ~12%, interconnect ~10%. Calling these models "compute-bound" would be wrong. The ~70% residual is almost certainly **prefill interleaved with decode** — at c=256 with ISL=1024 there are ~262K prompt tokens to process, and TTFT p99 climbing to 5,470 ms shows requests queueing behind that work — plus scheduler and framework overhead per step.

So the three tiers are limited by three different things: **Kimi-K3 by memory bandwidth**, and **Qwen / Llama by prefill throughput and scheduling**, with no hardware unit near its ceiling. Their distance from the roofline is therefore expected rather than a defect — a *pure-decode* roofline is the wrong yardstick for a **mixed prefill+decode** serving benchmark.

> **This last part is inference, not measurement.** The residual is what is left after subtracting four estimated costs; it is not a profiled breakdown, and the individual estimates carry their own error. Settling it properly needs a profiler trace, or a decode-only run (short ISL, or prefill and decode measured separately) to isolate the prefill contribution.

### Reading the comparison

- **8B vs 70B — tracks model size, roughly.** Raw throughput differs only 1.60x, but that hides the GPU count: **per GPU** it is 14,963 vs 1,168 tok/s, a **12.8x** gap against an 8.8x active-parameter ratio. So the 70B recovers most of what its size costs by using 8 GPUs; the residual ~1.5x beyond pure size scaling is TP communication, a larger KV cache per token, and lower per-GPU efficiency. That is the expected shape.
- **70B vs Kimi-K3 — does NOT track model size, and that is the finding.** Both are TP=8, so per GPU it is 1,168 vs 157 tok/s — **7.4x** apart for only a **1.2x** difference in *active* parameters (70 B vs ~84 B). Size does not explain it; **weight traffic** does. At batch 64 Kimi's tokens route independently across 896 experts, so 610 of them fire per layer and the engine reads **931 GB per step vs the 70B's 68 GB — 13.7x more traffic for 1.2x more active parameters.** A 7.4x slowdown from 13.7x more traffic is actually *better* than linear, because Kimi converts bandwidth to tokens more efficiently (29% of roofline vs 4%). This is the central cost of MoE: sparse activation saves FLOPs but not bytes, and bytes are the binding constraint. Full analysis in `kimi-k3.md`.

> **Caveat: different TP.** Tier 1 is TP=1 (single GPU), tiers 2 and 3 are TP=8. Throughput is therefore not normalized per GPU, and the tiers answer "what can this box serve for this model" rather than "which model is more efficient per GPU".

## Tier 1 — `Qwen3-8B-FP8` (TP=1)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.17 | 159.3 | 317.5 | 45.8 | 51.9 | 6.24 | 6.28 | 10 |
| 2 | 0.32 | 301.0 | 596.6 | 36.5 | 97.6 | 6.57 | 6.67 | 20 |
| 4 | 0.63 | 573.6 | 1,152.9 | 42.7 | 89.4 | 6.71 | 6.77 | 40 |
| 8 | 1.23 | 1,146.3 | 2,284.2 | 42.3 | 145.3 | 6.77 | 6.82 | 80 |
| 16 | 2.38 | 2,183.4 | 4,390.1 | 43.1 | 246.3 | 7.11 | 7.22 | 160 |
| 32 | 4.36 | 4,028.9 | 8,045.0 | 43.1 | 475.2 | 7.67 | 7.85 | 320 |
| 64 | 7.87 | 7,258.5 | 14,520.3 | 53.3 | 915.1 | 8.41 | 8.88 | 640 |
| 128 | 12.08 | 11,110.8 | 22,245.9 | 68.2 | 1,782.5 | 11.14 | 11.69 | 1,280 |
| 256 | 16.23 | 14,962.7 | 29,910.9 | 182.2 | 3,566.0 | 16.34 | 17.88 | 2,560 |

- Peak **14,962.7 tok/s** at concurrency 256.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 2.6x (6.24 -> 16.34 ms) while throughput grows 93.9x — the batching trade-off for this model.

## Tier 2 — `Llama-3.1-70B-Instruct-FP8` (TP=8)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.11 | 101.1 | 201.5 | 71.9 | 129.6 | 9.81 | 9.90 | 10 |
| 2 | 0.20 | 191.1 | 378.7 | 71.2 | 140.9 | 10.36 | 10.43 | 20 |
| 4 | 0.41 | 373.7 | 751.1 | 72.1 | 148.9 | 10.28 | 10.36 | 40 |
| 8 | 0.77 | 711.0 | 1,416.7 | 71.9 | 222.2 | 10.93 | 11.04 | 80 |
| 16 | 1.48 | 1,352.1 | 2,718.5 | 71.8 | 387.6 | 11.49 | 11.72 | 160 |
| 32 | 2.63 | 2,432.4 | 4,857.1 | 72.8 | 753.7 | 12.74 | 13.10 | 320 |
| 64 | 4.47 | 4,124.8 | 8,251.5 | 91.9 | 1,397.7 | 15.00 | 15.60 | 640 |
| 128 | 6.97 | 6,412.5 | 12,839.1 | 110.3 | 2,942.0 | 19.47 | 20.44 | 1,280 |
| 256 | 10.13 | 9,342.4 | 18,675.7 | 154.5 | 5,470.0 | 26.74 | 28.44 | 2,560 |

- Peak **9,342.4 tok/s** at concurrency 256.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 2.7x (9.81 -> 26.74 ms) while throughput grows 92.4x — the batching trade-off for this model.

## Tier 3 — `Kimi-K3` (TP=8)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 46.1 | 91.8 | 224.9 | 273.8 | 21.48 | 21.56 | 10 |
| 2 | 0.09 | 87.0 | 172.5 | 251.5 | 446.2 | 22.58 | 22.98 | 20 |
| 4 | 0.17 | 154.2 | 309.9 | 256.5 | 495.1 | 24.98 | 25.32 | 40 |
| 8 | 0.31 | 288.0 | 573.9 | 257.5 | 622.9 | 27.02 | 27.76 | 80 |
| 16 | 0.55 | 500.9 | 1,007.0 | 261.7 | 1,073.4 | 31.21 | 32.24 | 160 |
| 32 | 0.89 | 824.0 | 1,645.4 | 273.9 | 1,896.9 | 37.83 | 40.08 | 320 |
| 64 | 1.37 | 1,258.5 | 2,517.5 | 285.6 | 4,236.4 | 49.91 | 53.10 | 640 |

- Peak **1,258.5 tok/s** at concurrency 64.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 2.3x (21.48 -> 49.91 ms) while throughput grows 27.3x — the batching trade-off for this model.

## Metric definitions

| Metric | Meaning |
|---|---|
| TTFT | Time to first token — prefill latency; the user-perceived lag. |
| TPOT | Time per output token — steady-state decode speed after the first token. |
| output tok/s | Generated tokens/s across all concurrent requests. |
| total tok/s | Input + output tokens/s (prefill work included). |

## Caveats

- **The load generator is co-located with the server**, competing for host CPU. Standard ATOM/vLLM practice, but not a clean client/server split; req/s at high concurrency is mildly pessimistic.
- `--ignore-eos` forces exactly OSL tokens per request, so throughput is not skewed by early stopping — comparable, but not representative of real traffic.
- `--random-range-ratio 0.8` jitters prompt lengths so prefix caching cannot inflate results.
- KV cache dtype is fp8; Kimi-K3 additionally runs with prefix caching disabled (required — KDA recurrent state cannot be rebuilt from the paged cache).

## Deep dive

`kimi-k3.md` analyses tier 3 in detail: achieved TFLOP/s, the GPU memory breakdown (weights vs KV pool), why the workload is HBM-bandwidth-bound rather than compute- or interconnect-bound, and the intra-GPU vs intra-node data volumes.

## Source data

| What | Where |
|---|---|
| Per-concurrency JSON / logs | `logs/atom/sweep_*/c<N>.{json,log}` |
| Sweep summaries | `logs/atom/sweep_*/summary.txt` |
| Server logs | `logs/atom/server_*/atom_server.log` |
| This table as CSV | `results/atom.csv` |

