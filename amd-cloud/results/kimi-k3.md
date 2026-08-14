# Kimi-K3 on 8 × MI355X — compute and communication analysis

Serving `moonshotai/Kimi-K3` (2.78 T params, 1.5 TB MXFP4 checkpoint) on a single
8 × MI355X node via ATOM, TP=8. Measured 2026-08-14, Part D tier 3.

**Run configuration** (from the server log, not assumed):

| Setting | Value |
|---|---|
| Parallelism | `tensor_parallel_size=8`, PP=1, DP=1, **EP off** (`enable_expert_parallel=False`) |
| Quantization | routed MoE experts **MXFP4** (`mxfp4-pack-quantized`, group_size 32); attention/dense online-quantized to **PTPC-FP8** |
| KV cache dtype | fp8 |
| `max_model_len` / `max_num_seqs` | 16384 / 64 |
| `gpu_memory_utilization` | 0.93 |
| Prefix caching | **disabled** (required — KDA recurrent state can't be rebuilt from the paged cache) |
| Workload | ISL/OSL 1024/1024, `--ignore-eos`, concurrency 1→64 |

**Architecture** (parsed from `config.json`): 93 layers — **24 MLA full-attention** +
**69 KDA linear-attention**; hidden 7168; MoE with **896 routed experts, top-16 + 2 shared**,
expert hidden 3584 → 3072. Computed total ≈ **2.76 T params**, matching the advertised
2.78 T (validates the parse).

---

## Terminology — HBM and XGMI

Two different data paths, constantly confused, and the whole bottleneck analysis turns on
telling them apart.

**HBM — High Bandwidth Memory.** The GPU's own on-package memory ("VRAM"), where weights,
KV cache, and activations live. Stacked DRAM dies sitting on the same package as the compute
die, connected by an extremely wide bus (8192 bits on MI355X). Every time a kernel reads a
weight matrix, it reads it *from HBM*. This is the **intra-GPU** path — inside one GPU, no
other GPU involved.

| | MI355X |
|---|---|
| Capacity | 288 GB per GPU (2304 GB across 8) |
| Bandwidth | **8 TB/s per GPU** (8000 GB/s) |
| Type | HBM3E, on-package |
| Carries here | model weights, KV cache, activations |

**XGMI — the GPU↔GPU interconnect** (AMD Infinity Fabric; "xGMI" = inter-chip Global Memory
Interconnect). Dedicated high-speed links wired directly between GPUs so one GPU can read or
send data to another **without going through the CPU or PCIe**. This is the **intra-node**
path — between GPUs inside the same server. It is AMD's counterpart to NVIDIA's NVLink.

| | MI355X node |
|---|---|
| Topology | all-to-all mesh (K₈) — every GPU has a direct link to all 7 others, 1 hop |
| Per link | 153.6 GB/s bidirectional (76.8 GB/s each direction) |
| Per GPU aggregate | 1075 GB/s bidirectional (**~537 GB/s per direction**, ×7 links) |
| Carries here | activation all-reduces only (see §4) |

**Why the distinction decides everything.** HBM is ~15× faster than XGMI per GPU (8000 vs
537 GB/s), so the instinct is that HBM can never be the constraint. That is exactly backwards
for this workload: HBM has to move **~116 GB per step** (every activated expert's weights)
while XGMI moves **~0.3 GB** (a few activation vectors). Bandwidth ratio 15:1, traffic ratio
390:1 — so the *slower* link is the idle one. That inversion is the core finding of §3.

Two other paths appear in this report for contrast but carry no significant traffic here:

- **PCIe Gen5 ×16** (64 GB/s/direction) — host↔GPU. Used only to load the checkpoint from
  NVMe at startup (~4 min for 1.5 TB); idle during serving.
- **Inter-node network** — not applicable. This is a single-node run; nothing leaves the box.

---

## 0. Overview — the short version

**§1 Compute** — 1,259 tok/s at c=64 (27× scaling from c=1, TPOT only 2.3× worse). Achieved
**212 TFLOP/s aggregate = 26.5/GPU = 1.1% of BF16 peak**. Only 84 B of 2.78 T params activate
per token (3.0%).

**§2 Memory** — It loads far more than weights. Per GPU: **190.4 GB weights + 57.7 GB KV
pool** + 13.2 GB non-torch. The KV number decodes exactly:
13,824 B/token = `(kv_lora 512 + rope 64) × 1 B fp8 × 24 MLA layers`. That exact match proves
two things — **only the 24 full-attention layers use paged KV** (the 69 KDA layers keep
fixed recurrent state), and **KV is replicated across TP ranks, not sharded**. Capacity is
4.17 M tokens/GPU; the benchmark used **3.1%** of it.

**§3 Bottleneck — intra-GPU HBM bandwidth, decisively.** Compute 1.1% utilized, XGMI 1.1%,
**HBM ~29%** (2287 GB/s of 8000). The mechanism is a MoE subtlety worth highlighting: at
batch 64, tokens route independently, so **610 of 896 experts activate per layer**, not 16.
Weight traffic per step goes 3.4 GB → 116 GB. MXFP4 is load-bearing here — at BF16 this would
demand ~9.1 TB/s and exceed HBM outright. **~29% is not the ceiling**: a pure streaming read
(`babel`) reaches 91% on this box, and the gap is because each expert processes only ~1.7
tokens at batch 64 — a GEMV, which cannot saturate HBM. Raising `max-num-seqs` is the lever;
50–65% looks reachable, 90% does not.

**§4 Communication** — XGMI carries **only activations**: 186 all-reduces/token, 2.67 MB/token,
**5.87 GB/s per GPU (1.1% of ceiling)**. No all-to-all, because EP is off. No gradients
(inference), no KV exchange (replicated), no weight movement. HBM:XGMI ratio is roughly
**390:1**. This settles the Part B question directly — the N=5/6/7 cliff is irrelevant here,
and TP=8 is a power-of-2 arity that never triggers it anyway. **But 1.1% is a floor, not the
cost**: each call is only 14 KB–896 KB, far below the 16 MiB–8 GiB Part B swept, so these
collectives are latency-dominated, not bandwidth-dominated. With realistic small-message
latency the 186 serialized calls per step plausibly consume **~3–8% of step time**.

**§5** — EP looked like the top tuning lever (trading HBM traffic at 29% for all-to-all at
1%), so it was **tested — and it is not supported for this model**: ATOM raises
`NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel
masking`. MXFP4 experts and EP are mutually exclusive in this build, so the bandwidth
headroom is real but not exploitable by that route. The remaining lever is
`max_num_seqs=64`, which is the binding limit here — not hardware.

Two honesty notes: the KDA parameter count is a config-derived approximation (±15% on the
84 B figure, though the "~1% of peak" conclusion is unaffected by that margin), and all
derived volumes are computed from measured throughput plus parsed architecture — while the
entire memory table is read verbatim from the server's own budget line.

---

## 1. Computing performance

| Concurrency | Throughput (tok/s) | TTFT med (ms) | TPOT med (ms) | req/s |
|---:|---:|---:|---:|---:|
| 1 | 46.1 | 224.9 | 21.48 | 0.05 |
| 2 | 87.0 | 251.5 | 22.58 | 0.09 |
| 4 | 154.2 | 256.5 | 24.98 | 0.17 |
| 8 | 288.0 | 257.5 | 27.02 | 0.31 |
| 16 | 500.9 | 261.7 | 31.21 | 0.55 |
| 32 | 824.0 | 273.9 | 37.83 | 0.89 |
| 64 | **1258.5** | 285.6 | 49.91 | 1.37 |

Throughput scales **27×** from c=1 to c=64 (46 → 1259 tok/s) while TPOT grows only 2.3×
(21.5 → 49.9 ms) — batching is working well and the engine is nowhere near saturation at
c=64. The sweep stops at 64 because `max_num_seqs=64`; past that you would measure queueing,
not the engine.

### Achieved TFLOP/s

MoE is sparse, so only **top-16 + 2 shared of 896** experts fire per token. Active params per
token ≈ **84 B** (of 2.76 T total — 3.0% activation ratio):

| Component | Active params/token |
|---|---:|
| FFN (16 routed + 2 shared experts × 92 layers) | 55.4 B |
| Attention (24 MLA + 69 KDA) | 27.8 B |
| Embedding / lm_head | 1.2 B |

At 2 FLOP per active param per token:

| Concurrency | Aggregate TFLOP/s | Per GPU | % of MI355X BF16 peak (2500) |
|---:|---:|---:|---:|
| 1 | 7.8 | 0.97 | 0.04% |
| 64 | **212.4** | **26.5** | **1.1%** |

**Decode is nowhere near compute-bound** — barely 1% of peak matrix throughput. That is
expected and not a defect: autoregressive decode issues one token per sequence per step, so
every weight matrix is used for a single narrow GEMV-like operation. This is a
*memory-bandwidth* regime, quantified in §3.

> Caveat: the KDA (linear-attention) parameter count is an approximation from the config
> dimensions (q/k/v/gate projections at 96 heads × 128 head_dim); the MoE and MLA terms are
> exact. KDA dominates the attention figure, so treat the 84 B active-param number as ±15%.
> The conclusion (decode is ~1% of peak) is far too large a margin to be affected.

---

## 2. GPU memory usage

Measured per rank at load time, straight from the server log:

```
total_gpu=287.98GB  utilization=0.93  budget=267.83GB
peak_torch=190.36GB  non_torch=13.16GB  cudagraph_est=0.84GB  safety=5.76GB
available_for_kv=57.70GB  block_bytes=1769472  num_kvcache_blocks=32980
```

Per GPU (× 8 for the node):

| Component | Per GPU | Node total | What it is |
|---|---:|---:|---|
| Model weights + framework (`peak_torch`) | **190.4 GB** | 1523 GB | The TP=8 weight shard, dominated by MXFP4 experts |
| Non-torch (RCCL buffers, HIP runtime, driver) | 13.2 GB | 105 GB | |
| CUDA-graph pool | 0.84 GB | 6.7 GB | 8 captured batch sizes (1→64) |
| Safety margin | 5.8 GB | 46 GB | |
| **KV cache pool** | **57.7 GB** | **462 GB** | Allocated up front from what's left |
| Unused (above 0.93 util) | ~20 GB | ~160 GB | Headroom deliberately not claimed |

**It does not load only weights.** Weights are ~190 GB/GPU (the 1.5 TB checkpoint sharded 8
ways ≈ 190 GB, consistent), and a **57.7 GB KV pool per GPU** is carved out on top —
462 GB node-wide, about **20% of the box's 2304 GB HBM** dedicated to KV.

### KV cache details

Measured `block_bytes / block_tokens = 1769472 / 128` = **13,824 B per token per GPU**.

That figure is itself informative. MLA stores a *compressed latent* KV: `kv_lora_rank(512) +
qk_rope_head_dim(64)` = 576 values/token/layer, at fp8 = 576 B, × **24 full-attention
layers** = 13,824 B — an exact match. Two conclusions follow:

1. **Only the 24 MLA layers consume paged KV.** The 69 KDA layers keep a fixed-size
   *recurrent state* per request instead of a growing cache — that's the whole point of
   linear attention, and it's why a 93-layer model has the KV footprint of a 24-layer one.
2. **The KV cache is replicated, not sharded, across TP ranks.** Each rank holds the full
   13,824 B/token. MLA's latent is shared across heads, so sharding it would require
   re-gathering it every step. Replication trades ~50 GB of otherwise-idle HBM for zero
   communication.

Capacity: 57.7 GB ÷ 13,824 B = **4.17 M tokens** per GPU. The benchmark used
64 seqs × ~2048 tokens = **1.81 GB, only 3.1% of the pool**. KV was never remotely a
constraint here — this config could serve ~2000 concurrent 2K-token sequences on memory
alone, far past the `max_num_seqs=64` limit.

---

## 3. What is the bottleneck?

**Intra-GPU HBM bandwidth — decisively.** Not compute, not the interconnect.

| Resource | Demand at c=64 | MI355X capability | Utilization |
|---|---:|---:|---:|
| Compute | 26.5 TFLOP/s per GPU | 2500 TFLOP/s BF16 | **1.1%** |
| **HBM bandwidth** | **~2287 GB/s per GPU** | 8000 GB/s | **~29%** |
| XGMI (GPU↔GPU) | 5.9 GB/s per GPU | ~537 GB/s (1-direction aggregate) | **~1.1%** |

Compute and interconnect are both ~1% utilized; HBM traffic is two orders of magnitude
closer to its ceiling. The ranking is unambiguous even with generous error bars.

### 3.1 Why HBM — the MoE batching mechanism

At **batch 1**, only 16+2 experts fire per layer → ~3.4 GB/GPU of weights read per step. At
**batch 64**, the 64 tokens route independently, so the expected number of *distinct* experts
touched per layer is `E × (1 − (1−1/E)^(B·topk))` = **610 of 896** — not 16. Weight reads
jump to **116 GB/GPU per step**, which is the ~2287 GB/s in the table above.

This is the defining property of sparse MoE: **compute grows with batch, but weight traffic
grows much faster** until nearly every expert is touched every step.

MXFP4 is what makes this tractable at all. At BF16 the same reads would be **4× larger**
(~9.1 TB/s demanded), exceeding HBM bandwidth outright and making the model
bandwidth-*starved* rather than merely bandwidth-dominated.

### 3.2 How to improve it — raising HBM utilization above the current ~28%

The bottleneck is HBM, but the box is **not** delivering all the HBM it has: 2287 of
8000 GB/s. So the optimization target is clear — **close the gap between 28% and what the
memory system can actually sustain.** Two questions follow: what is the real ceiling, and
what moves us toward it.

**What "full" means here.** RVS `babel` — a pure streaming-read kernel with no compute, no
routing, no attention — measured **7,260 GB/s = 91% of the 8 TB/s spec** on this same box
(Part A, `logs/rvs/health_*/babel.log`). So **~91% is the practical hardware ceiling**, and
the engine is currently delivering roughly **one third of what the memory system can do**.

**Why the shortfall: the MoE GEMMs are too thin.** With ~610 experts fired across 64 tokens,
each expert sees only:

```
tokens per expert = B x topk / distinct_experts = 64 x 16 / 610 = 1.7
```

Each expert's ~16.5 MB weight matrix is read to do arithmetic on **1.7 tokens** — a
matrix-*vector* product, not matrix-matrix. A GEMV cannot issue enough concurrent memory
requests to saturate HBM: it is memory-bound *and* latency-bound simultaneously. That is
where the missing ~62 percentage points go — not to any other consumer.

**The lever is batch size, and MoE has a favourable asymmetry.** Past a certain batch every
expert activates anyway, so weight bytes **plateau** while tokens keep growing — each weight
read amortizes over more work:

| Batch | Experts fired/layer | Tokens per expert | Weights read/GPU | KV used/GPU |
|---:|---:|---:|---:|---:|
| 64 *(measured today)* | 610 | **1.7** | 116 GB | 1.8 GB |
| 256 | 887 | 4.6 | 169 GB | 7.2 GB |
| 512 | 896 (all) | 9.1 | 170 GB | 14.5 GB |
| 1024 | 896 (all) | **18.3** | **171 GB** (flat) | 29.0 GB |

From batch 512 upward weight traffic is **constant at ~170 GB/GPU** — every additional token
is nearly free in bandwidth terms. Projecting with plausible utilization gains as the GEMMs
widen:

| Batch | Assumed HBM util | Step time | Projected tok/s |
|---:|---:|---:|---:|
| 64 *(measured)* | **28.5%** | 50.9 ms | **1,259** |
| 256 | ~45% | ~47 ms | ~5,500 |
| 1024 | ~60% | ~36 ms | ~29,000 |

**The memory is already allocated**: at batch 1024 the KV cache needs 29 GB/GPU against the
**57.7 GB pool** carved out at startup — over half of it sits idle today (§2). The binding
limit is **`--max-num-seqs 64`** inherited from the recipe, not hardware.

**Could it reach 90%?** Realistically no. 91% is what a pure streaming read achieves with
nothing else in the loop. A real engine must also do MXFP4 dequantization, expert
gather/scatter routing, MLA and KDA attention, 186 all-reduces per token, and interleaved
prefill — all of which consume step time without reading weights. **50–65% is the plausible
target**, which on the projection above is still a ~4–20× throughput gain over today.

Ranked, the ways to raise HBM utilization:

1. **Raise `--max-num-seqs`** (64 → 256 or higher). Biggest lever, costs nothing, memory
   already provisioned. Untested only because the recipe's value was taken as given.
2. **Speculative decoding / MTP** — verifies several tokens per weight read, which widens
   the GEMM exactly like a larger batch does.
3. **Expert parallelism** — would make each GPU read fewer, whole experts. **Tested and
   unavailable**: ATOM raises `NotImplementedError` for EP with the MXFP4 SiTUv2 kernel
   (§5, item 1).
4. **Prefill/decode disaggregation** — removes interleaved prefill from the decode step so
   the measured window is closer to pure weight streaming.

> The projected rows are arithmetic under an *assumed* utilization curve, not measurements —
> the batch/expert/byte columns are exact, the utilization guesses are not. Testing is cheap
> and is the obvious next experiment: re-run tier 3 with `MAX_NUM_SEQS=256` and `KIMI_CONC`
> extended past 64, then read the real numbers off the sweep.

### 3.3 Why TTFT is flat but TPOT rises

TTFT stays ~225→286 ms across a 64× concurrency increase because prefill is genuinely
compute-dense (1024 tokens/request in parallel) and has compute headroom. TPOT rises 2.3×
because decode adds weight-read traffic per step as more experts activate. The two metrics
sit in different regimes — further confirmation that decode is bandwidth-limited.

---

## 4. Data communication analysis

### Intra-node GPU↔GPU (XGMI) — activations only

With **TP=8 and EP disabled**, every expert is sharded across all 8 GPUs, so there is **no
expert-routing all-to-all**. The only cross-GPU traffic is TP activation reduction:

| Property | Value |
|---|---|
| Collective | **all-reduce** (RCCL), 2 per layer — after attention out-proj, after FFN down-proj |
| Count per token | 2 × 93 = **186 all-reduces** |
| Payload per call per token | `hidden_size × 2 B` (bf16) = **14.0 KB** |
| Payload per token (all layers) | **2.67 MB** |

| Concurrency | Steps/s | Payload/step | Wire bytes/step¹ | Sustained per GPU |
|---:|---:|---:|---:|---:|
| 1 | 46.1 | 2.7 MB | 4.7 MB | **0.22 GB/s** |
| 64 | 19.7 | 170.7 MB | 298.6 MB | **5.87 GB/s** |

¹ busbw convention: an all-reduce moves `2(N−1)/N × payload` on the wire.

At **5.87 GB/s against a ~537 GB/s per-direction ceiling (~1.1%)**, the interconnect is
almost entirely idle. This directly answers the question Part B raised: **the N=5/6/7 RCCL
cliff is irrelevant to this workload.** Even the degraded ~45 GB/s cliff bandwidth is ~8×
more than this needs — and TP=8 is a power-of-2 arity that never hits the cliff anyway.

**What is NOT transferred over XGMI:** weights (static, resident per GPU), KV cache
(replicated, never exchanged), gradients (inference — none exist), and expert tokens (EP is
off; no all-to-all).

#### The message-size regime — why "1% utilized" understates the cost

Aggregate bandwidth is the wrong lens here, and it is worth being precise about why.

| | Per all-reduce call |
|---|---|
| Payload at c=1 | **14.0 KB** |
| Payload at c=64 | **896 KB** |
| Wire bytes at c=64 (`2(N−1)/N`) | 1.53 MB |

Part B's rccl-tests sweep measured **16 MiB → 8 GiB**, where all-reduce reaches 396.6 GB/s
busbw (74% of spec). **Every call in this workload is 1–3 orders of magnitude smaller than
the smallest message Part B measured.** At 14 KB–896 KB, a collective is in the
*latency-dominated* regime — fixed per-call overhead (kernel launch, ring setup, barrier)
dwarfs the transfer itself. Part B's busbw numbers therefore say almost nothing about what
this workload experiences, in either direction.

A step-time budget at c=64 (step = 50.9 ms, 186 all-reduces):

| Assumption | All-reduce time/step | Share of step |
|---|---:|---:|
| Pure bandwidth, zero overhead | 0.56 ms | 1.1% |
| + 5 µs fixed latency per call | 1.49 ms | 2.9% |
| + 10 µs fixed latency per call | 2.42 ms | 4.7% |
| + 20 µs fixed latency per call | 4.28 ms | 8.4% |

So the honest range is **~1% (bandwidth only) to ~8% (with realistic small-message
latency)** — the "1.1% utilized" figure is a floor, not an estimate of cost. The multiplier
is the call *count*: 186 serialized collectives per step means even a few microseconds each
compounds into milliseconds. This is precisely the mechanism that could make EP
counterproductive despite abundant spare bandwidth (see §6).

> Not directly measured: we have no per-call RCCL timing from this run (no profiler trace),
> so the latency rows above are illustrative arithmetic over a plausible range, not
> measurements. Confirming the real figure needs `--profile` or `NCCL_DEBUG=INFO` timing —
> worth doing if collective cost ever becomes the thing to optimize.

#### How the traffic maps onto the mesh

The K₈ topology means every rank has a direct 1-hop link to all 7 others, so a ring
all-reduce at N=8 can use all 7 links concurrently and no traffic is ever relayed. Two
consequences specific to this run:

- **TP=8 uses the whole mesh.** Both ring phases (reduce-scatter then all-gather) spread
  across all 7 links per GPU, which is why the effective ceiling is the ~537 GB/s aggregate
  rather than a single link's 76.8 GB/s.
- **NUMA is irrelevant to these transfers.** GPUs 0–3 sit on NUMA node 0 and 4–7 on node 1,
  but every GPU pair is still 1 XGMI hop; the cross-socket boundary only costs when host
  memory is in the path, which it is not for GPU↔GPU all-reduce.

#### Bidirectional vs per-direction

Spec sheets quote **1075 GB/s per GPU bidirectional**; the useful number for a ring
all-reduce is the **~537 GB/s per-direction** figure, since each ring phase pushes data one
way around the ring. All utilization percentages in this report use the per-direction
number — the conservative choice. Against the bidirectional headline the same traffic would
look half as significant (~0.55%), which would overstate the headroom.

### Intra-GPU (HBM) — dominated by weights

Per decode step at c=64, per GPU:

| Traffic | Bytes/step | Share |
|---|---:|---:|
| **Expert weights (MXFP4)** | **~116 GB** | **~98%** |
| Attention + dense weights (PTPC-FP8) | ~2 GB | ~2% |
| KV cache read (64 seqs × ~2048 tok × 13.8 KB) | ~1.8 GB | ~1.5% |
| Activations (read+write, 170 MB payload) | ~0.3 GB | <1% |

The asymmetry is stark: **HBM moves ~116 GB/step while XGMI moves ~0.3 GB/step — roughly
390:1.** Any optimization effort belongs on the memory side.

---

## 5. Further discussion

**1. EP is off — and it turns out it *cannot* be turned on for this model. Tested.**
`enable_expert_parallel=False` means TP shards every expert across 8 GPUs, so each GPU reads
a slice of *every* activated expert. EP would instead place whole experts on specific ranks:
fewer, complete expert reads per GPU (less HBM traffic) paid for with **all-to-all token
routing** over the near-idle XGMI. With HBM at ~29% and XGMI at ~1%, the trade looked
favorable on paper, so it was the obvious first experiment.

**It was run, and it fails at model load** (`logs/atom/kimi_ep_ab_20260814_183129/`):

```
NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel masking.
```

This is not a misconfiguration — ATOM raises it deliberately. The **MXFP4 SiTUv2 grouped-MoE
kernel** (the gfx950-native fast path that makes this model viable at all, per
`ATOM/recipes/Kimi-K3.md`) does not implement the expert-parallel masking variant. The
combination is simply not built: MXFP4 experts and EP are **mutually exclusive** in this
ATOM build.

So the HBM-vs-XGMI trade cannot be evaluated on this model today, and the bandwidth headroom
identified in §3 is **not currently exploitable by this route**. The realistic paths forward
are (a) an ATOM/AITER release that adds EP masking to the MXFP4 SiTU path, or (b) accepting
the a16w4 dequant penalty by running experts at a wider dtype where EP kernels do exist —
which would raise HBM traffic 4×, defeating the purpose. This lands as a genuine software
limitation rather than a tuning oversight, and it is worth re-testing on future image
versions.

**2. `max_num_seqs=64` is the binding limit, not hardware.** KV was 3.1% used and compute
~1%. The recipe's value is conservative; raising it (with `max_num_batched_tokens`) should
push throughput substantially before any hardware limit appears. Expect diminishing returns
once all 896 experts activate per step (~batch 128), where weight traffic plateaus.

**3. Prefix caching is disabled for correctness, and it costs real throughput.** KDA's
recurrent state is per-request and cannot be reconstructed from the paged MLA cache. In
workloads with shared prefixes (system prompts, few-shot), this forfeits a large win that
non-KDA models get for free — an architectural trade, not a tuning oversight.

**4. The hybrid attention design is what makes 2.78 T fit.** Only 24 of 93 layers keep a
growing KV cache; 69 use fixed-size KDA state. A conventional 93-layer model would need
~4× the KV per token, and the 4.17 M-token capacity would collapse to ~1 M.

**5. Load time was ~4 minutes** for 1.5 TB from NVMe (~6 GB/s effective), plus ~10 s of CUDA
graph capture across 8 batch sizes. Well under the 40-minute timeout budgeted.

**6. Comparison across Part D tiers** — TP=8 costs surprisingly little:

| Model | Params | TP | Peak tok/s | TPOT @ c=1 |
|---|---:|---:|---:|---:|
| Qwen3-8B-FP8 | 8 B | 1 | 14,963 | 6.24 ms |
| Llama-3.1-70B-FP8 | 70 B | 8 | 9,342 | 9.81 ms |
| **Kimi-K3** | **2.78 T** (84 B active) | 8 | **1,259** | 21.48 ms |

Kimi-K3 is ~350× larger than the 8B by total params but only ~12× slower — because only 3%
of it activates per token. Against Llama-70B (comparable *active* size, 84 B vs 70 B) it is
7.4× slower, which is the real cost of MoE: sparse activation saves FLOPs but not weight
*traffic*, and traffic is the binding constraint.

---

## Source data

| What | Where |
|---|---|
| Sweep summary + per-concurrency JSON | `logs/atom/sweep_20260814_164903/` |
| Server log (memory budget, engine config) | `logs/atom/server_20260814_164506/atom_server.log` |
| Model config | `/mnt/scratch/shaohao/models/Kimi-K3/config.json` |
| Cross-tier table | `results/atom.md` |

Derived figures (active params, FLOP/s, HBM and XGMI volumes) are computed from the measured
throughput and the parsed architecture; the memory table is read directly from the server's
own budget line.
