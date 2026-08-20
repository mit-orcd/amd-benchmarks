# Kimi-K3 on 8 × MI355X — max-num-seqs experiment (compute and communication analysis)

Serving `moonshotai/Kimi-K3` (2.78 T params, 1.5 TB MXFP4 checkpoint) on a single
8 × MI355X node via ATOM, TP=8, with **`--ISL raised 1024 -> 4096 (cap 256)** to
test whether the in-flight cap — not hardware — was the binding limit.
Source run: `kimi_isl4096_20260820_131217`.

> This file reports **only** the raised-`max-num-seqs` run. Baseline runs are in
> `kimi-k3-base.md` (original recipe, max-num-seqs=64) and `kimi-k3-mad.md` (MAD recipe,
> max-num-seqs=64); their head-to-head is `kimi-k3-comparison.md`. All are kept
> separate so none overwrites another.

**Run configuration** (from the server log and launch command, not assumed):

| Setting | Value |
|---|---|
| Image | `rocm/atom-dev:latest` (the better-performing baseline, see kimi-k3-comparison.md) |
| Parallelism | `tensor_parallel_size=8`, PP=1, DP=1, EP off |
| Quantization | MXFP4 routed experts + PTPC-FP8 via `--online_quant_config` (original recipe) |
| KV cache dtype | fp8 |
| `max_model_len` / `max_num_seqs` | 16384 / **256** (raised from 64) |
| `max_num_batched_tokens` | 16384 |
| `gpu_memory_utilization` | 0.93 |
| Prefix caching | disabled (KDA recurrent state cannot be rebuilt from paged cache) |
| Workload | ISL/OSL 1024/1024, `--ignore-eos`, concurrency 64→256 |
| Kernel-selection env vars | none (original recipe; MAD vars measured ~9% slower) |

**Architecture**: 93 layers — **24 MLA full-attention** + **69 KDA linear-attention**;
hidden 7168; MoE with **896 routed experts, top-16 + 2 shared**, expert hidden
3584 → 3072. ≈ **2.76 T params** total, ~84 B active per token (3.0%).

---

## 0. Overview — the short version

**§1 Compute** — **2,098.2 tok/s** peak at c=256 (TPOT 120.03 ms, TTFT 531.6 ms). Achieved **352.5 TFLOP/s aggregate = 44.1/GPU = 1.8% of BF16 peak**.

**§2 Memory** — per GPU: **190.6 GB weights** + **57.0 GB KV pool**. KV is 13.5 KB/token — only the 24 MLA layers keep a paged cache; the 69 KDA layers hold fixed recurrent state.

**§3 Bottleneck — intra-GPU HBM bandwidth.** Compute 1.8% utilized, XGMI 1.8%, **HBM ~17%** (1,383.4 GB/s of 8,000). At batch 256, **887 of 896 experts** activate per layer, so weight traffic is 168.8 GB/GPU/step.

**§4 Communication** — XGMI carries only activations: 186 all-reduces/token, **9.79 GB/s per GPU (1.8% of ceiling)**. No all-to-all (EP off), no gradients, no KV exchange.

---

## 1. Computing performance

**Concurrency** = number of independent requests served *at once*, each with its own
prompt and growing output. It is a client-side load setting, not a hardware unit — all
requests are batched together on the **same 8 GPUs** (TP=8), not one per GPU.

| Concurrency | Throughput (tok/s) | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | req/s | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1,225.3 | 318.6 | 15,909.6 | 50.08 | 1.33 | 640 |
| 128 | 1,671.0 | 471.9 | 24,055.1 | 75.01 | 1.81 | 1,280 |
| 256 | **2,098.2** | 531.6 | 47,659.8 | 120.03 | 2.28 | 2,560 |

Throughput scales **1.7×** from c=64 to c=256 while TPOT grows 2.4×.

With `--max-num-seqs 256`, concurrency up to 256 is admitted to the batch rather than
queued. Compare against `kimi-k3-mad.md`, where the same concurrencies ran against
a 64-slot server and TTFT median reached 150 s at c=256 while throughput stayed flat.

### Achieved TFLOP/s

MoE is sparse: only **top-16 + 2 shared of 896** experts fire per token, so active
params ≈ **84 B** of 2.76 T (3.0%). At 2 FLOP per active param per token:

| Concurrency | Aggregate TFLOP/s | Per GPU | % of BF16 peak (2500) |
|---:|---:|---:|---:|
| 64 | 205.8 | 25.7 | 1.03% |
| 128 | 280.7 | 35.1 | 1.40% |
| 256 | 352.5 | 44.1 | 1.76% |

**Decode is nowhere near compute-bound** — a fraction of peak matrix throughput. That
is expected: autoregressive decode does one token per sequence per step, so every
weight matrix serves a narrow GEMV-like operation. This is a *memory-bandwidth*
regime, quantified in §3.

> The KDA (linear-attention) parameter count is approximated from config dimensions;
> the MoE and MLA terms are exact. Treat the 84 B active figure as ±15%. The
> conclusion (decode is ~1% of peak) has far too much margin to be affected.

## 2. GPU memory usage

Measured per rank at load time, from the server's own budget line:

```
total_gpu=287.98GB  utilization=0.93  budget=267.83GB
peak_torch=190.59GB  non_torch=13.64GB  cudagraph_est=0.88GB  safety=5.76GB
available_for_kv=56.95GB  block_bytes=1769472  num_kvcache_blocks=26430
```

| Component | Per GPU | Node total (×8) |
|---|---:|---:|
| Model weights + framework (`peak_torch`) | 190.6 GB | 1,525 GB |
| Non-torch (RCCL buffers, HIP runtime) | 13.6 GB | 109 GB |
| CUDA-graph pool | 0.88 GB | 7.0 GB |
| Safety margin | 5.76 GB | 46.1 GB |
| **KV cache pool** | **57.0 GB** | **456 GB** |

**It does not load only weights** — a KV pool is carved out on top.

### KV cache details

`block_bytes / 128 tokens` = **13,824 B per token per GPU** (13.5 KB).

That decodes exactly: MLA stores a compressed latent KV of `kv_lora(512) + qk_rope(64)` = 576 values/token/layer at fp8 = 576 B, × **24 full-attention layers** = 13,824 B. Two conclusions follow:

1. **Only the 24 MLA layers consume paged KV.** The 69 KDA layers keep a
   fixed-size recurrent state per request instead — which is why a 93-layer model
   has the KV footprint of a 24-layer one.
2. **KV is replicated across TP ranks, not sharded.** MLA's latent is shared across
   heads, so sharding would force a re-gather every step. Replication trades idle
   HBM for zero communication.

Capacity: 57.0 GB ÷ 13,824 B = **4.12 M tokens** per GPU. At c=256 × ~2048 ctx that is 7.25 GB — **12.7% of the pool**.

## 3. What is the bottleneck?

**Intra-GPU HBM bandwidth.** Not compute, not the interconnect.

| Resource | Demand at c=256 | MI355X capability | Utilization |
|---|---:|---:|---:|
| Compute | 44.1 TFLOP/s per GPU | 2,500 TFLOP/s BF16 | **1.8%** |
| **HBM bandwidth** | **1,383.4 GB/s per GPU** | 8,000 GB/s | **~17%** |
| XGMI (GPU↔GPU) | 9.79 GB/s per GPU | ~537.6 GB/s (1-direction) | **~1.8%** |

### 3.1 Why HBM — the MoE batching mechanism

| Batch | Experts fired/layer | Tokens per expert | Weights read/GPU |
|---:|---:|---:|---:|
| 64 | 610 | 1.7 | 116.3 GB |
| 128 | 805 | 2.5 | 153.3 GB |
| 256 | 887 | 4.6 | 168.8 GB |

This is the defining property of sparse MoE: **compute grows with batch, but weight
traffic grows much faster** until nearly every expert is touched every step. MXFP4 is
what makes it tractable — at BF16 the same reads would be 4× larger and exceed HBM
bandwidth outright.

### 3.2 How to improve it — raising HBM utilization above the current ~17%

RVS `babel`, a pure streaming-read kernel with no compute or routing, measured
**7,260 GB/s = 91% of spec** on this box (Part A). So ~91% is the practical hardware ceiling, and this run delivers **19% of what the memory system can actually do**.

The shortfall is that MoE GEMMs are thin: at batch 256 each activated expert sees only
**4.6 tokens**, making each weight read a matrix-*vector*
product that cannot issue enough concurrent memory requests to saturate HBM.

Ranked levers:

1. **Raise `--max-num-seqs`** — biggest lever; weight traffic plateaus once all 896
   experts activate, so extra tokens become nearly free in bandwidth terms.
2. **Speculative decoding / MTP** — verifies several tokens per weight read.
3. **Expert parallelism** — unavailable: ATOM raises `NotImplementedError` for EP with
   the MXFP4 SiTUv2 kernel.
4. **Prefill/decode disaggregation** — removes interleaved prefill from decode steps.

**Could it reach 90%?** No. That is a pure streaming read with nothing else in the
loop. A real engine also does MXFP4 dequant, expert routing, MLA+KDA attention,
186 all-reduces per token, and interleaved prefill. **50–65% is the plausible
target.**

### 3.3 Why TTFT and TPOT behave differently

TTFT moves from 318.6 → 531.6 ms across the sweep while TPOT moves
50.08 → 120.03 ms. Prefill is compute-dense and has headroom;
decode adds weight-read traffic per step as more experts activate. The two metrics sit
in different regimes — further confirmation that decode is bandwidth-limited.

## 4. Data communication analysis

### Intra-node GPU↔GPU (XGMI) — activations only

With **TP=8 and EP disabled**, every expert is sharded across all 8 GPUs, so there is
**no expert-routing all-to-all**. The only cross-GPU traffic is TP activation reduction:

| Property | Value |
|---|---|
| Collective | **all-reduce** (RCCL), 2 per layer × 93 layers |
| Count per token | **186 all-reduces** |
| Payload per call per token | `hidden_size × 2 B` = **14.0 KB** |
| Payload per token (all layers) | **2.67 MB** |

| Concurrency | Steps/s | Payload/step | Wire bytes/step¹ | Sustained per GPU |
|---:|---:|---:|---:|---:|
| 64 | 19.1 | 170.7 MB | 298.6 MB | **5.72 GB/s** |
| 128 | 13.1 | 341.3 MB | 597.3 MB | **7.80 GB/s** |
| 256 | 8.2 | 682.6 MB | 1194.6 MB | **9.79 GB/s** |

¹ busbw convention: an all-reduce moves `2(N−1)/N × payload` on the wire.

At **9.79 GB/s against a ~537.6 GB/s per-direction ceiling (~1.8%)**, the interconnect is almost entirely idle. The
N=5/6/7 RCCL cliff found in Part B is irrelevant here — TP=8 is a power-of-2 arity that
never triggers it, and even the degraded cliff bandwidth would be ample.

**Message-size caveat**: each call is only 3584 KB at c=256 — far below the
16 MiB–8 GiB range Part B swept, so these collectives are **latency-dominated, not
bandwidth-dominated**. The 186 serialized calls per step mean the true cost is
higher than the raw utilization figure suggests.

> **Superseded (2026-08-20).** Measurement replaced this estimate: collectives are **0.8%** of host step time, the smallest bucket — not several percent. Tensor copies (41.3%) and KDA attention (38.6%) dominate. See `kimi-k3-profile.md`.


**What is NOT transferred over XGMI:** weights (resident per GPU), KV cache
(replicated), gradients (inference), expert tokens (EP off).

### Intra-GPU (HBM) — dominated by weights

Per decode step at c=256, per GPU:

| Traffic | Bytes/step | Share |
|---|---:|---:|
| **Expert weights (MXFP4)** | **~168.8 GB** | **~96%** |
| KV cache read | ~7.25 GB | ~4.1% |
| Activations | ~0.68 GB | ~0.4% |

HBM moves ~168.8 GB/step while XGMI moves ~1.19 GB/step — roughly
**141:1**. Optimization effort belongs on the memory side.

## 5. Further discussion

**1. EP is off and cannot be enabled for this model.** ATOM raises
`NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel
masking` — the MXFP4 SiTUv2 grouped-MoE kernel has no expert-parallel variant. MXFP4
experts and EP are mutually exclusive **on `rocm/atom-dev:latest`, the image the EP
test actually ran on** (2026-08-14). Not retested on the MAD-pinned image, so treat
it as image-specific rather than universal. The HBM-vs-XGMI trade cannot be
evaluated on this model today.

**2. `max_num_seqs` was raised to 256 for this run** — the whole point of the
experiment. Whether it moved throughput is answered by the §1 table versus the
~1,180 tok/s plateau the 64-slot runs hit; see `kimi-k3-comparison.md` §2.

**3. Prefix caching is disabled for correctness.** KDA recurrent state is per-request
and cannot be reconstructed from the paged MLA cache. In workloads with shared prefixes
this forfeits a win non-KDA models get for free — an architectural trade, not an
oversight.

**4. The hybrid attention design is what makes 2.78 T fit.** Only 24 of 93 layers keep
a growing KV cache; 69 use fixed-size KDA state. A conventional 93-layer model would
need ~4× the KV per token.

**5. This run used AMD's validated kernel selection.** The MAD env vars
(`ATOM_USE_TRITON_GEMM=1`, `ATOM_USE_TRITON_MOE=0`, `AITER_FLYDSL_FORCE=1`,
`ATOM_USE_UNIFIED_ATTN=1`, `ATOM_FORCE_ATTN_TRITON=1`, …) select the kernel set AMD
benchmarked and published against — unlike the earlier run, which used generic
defaults. Comparison between the two is in `notes-kimi-k3.md`.

## Source data

| What | Where |
|---|---|
| Per-concurrency JSON / logs | `logs/atom/kimi_isl4096_20260820_131217/c<N>.{json,log}` |
| Sweep summary | `logs/atom/kimi_isl4096_20260820_131217/summary.txt` |
| Server log (memory budget, engine config) | `logs/atom/kimi_isl4096_20260820_131217/atom_server.log` |
| Driver state | `logs/atom/kimi_isl4096_20260820_131217/STATE.txt` |
| This table as CSV | `results/kimi-k3-isl4096.csv` |
| Rerun rationale + config diff | `notes-kimi-k3.md` |
| Original (non-MAD) run | `kimi-k3-base.md` |

---

## Terminology — HBM and XGMI

**HBM — High Bandwidth Memory.** The GPU's own on-package memory, where weights, KV
cache and activations live. **8 TB/s per GPU**, 288 GB capacity. This is the
**intra-GPU** path — inside one GPU, no other GPU involved.

**XGMI — the GPU↔GPU interconnect** (AMD Infinity Fabric). Direct links between GPUs,
bypassing CPU and PCIe. All-to-all mesh (K₈), every pair 1 hop, **~537 GB/s per
direction** aggregate per GPU. This is the **intra-node** path; AMD's NVLink counterpart.

HBM is ~15× faster than XGMI per GPU, so the instinct is that HBM can never be the
constraint. For this workload that is backwards: HBM moves ~168.8 GB/step while XGMI
moves ~1.19 GB — the *slower* link is the idle one.

