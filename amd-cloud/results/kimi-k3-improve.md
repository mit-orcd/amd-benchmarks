# Kimi-K3 on 8 × MI355X — consolidated analysis (v1)

Serving `moonshotai/Kimi-K3` (2.78 T params, 1.5 TB MXFP4 checkpoint) via ATOM at TP=8 on a
single 8 × MI355X node. This file consolidates **three runs** into one narrative. The
per-run reports remain unchanged as sources of record:

| Run | Config | Detail file |
|---|---|---|
| **A — original** | `rocm/atom-dev:latest`, ATOM in-repo recipe, `max-num-seqs 64` | `kimi-k3-base.md` |
| **B — AMD MAD** | MAD-pinned image + 11 kernel env vars, `max-num-seqs 64` | `kimi-k3-mad.md` |
| **C — raised cap** | Run A's image/recipe, **`max-num-seqs 256`** | `kimi-k3-maxseqs.md` |

Run-B-vs-Run-A detail also lives in `kimi-k3-comparison.md`; rerun rationale and the `fla`
diagnosis in `notes-kimi-k3.md`.

---

## 1. Model and platform (constant across all three runs)

**Architecture** (parsed from `config.json`): 93 layers — **24 MLA full-attention + 69 KDA
linear-attention**; hidden 7168; MoE with **896 routed experts, top-16 + 2 shared**, expert
hidden 3584 → 3072. Total ≈ **2.76 T params**, of which only **~84 B (3.0%) activate per
token**. That sparsity is the single most important fact for reading everything below.

**Platform**: 8 × MI355X (gfx950), ROCm 7.14. Per GPU: 288 GB HBM at **8 TB/s**; XGMI
all-to-all mesh, ~537 GB/s per direction aggregate.

**Memory footprint** (identical across runs, read from the server's own budget line):

| Component | Per GPU | Node (×8) |
|---|---:|---:|
| Model weights + framework | 190.4 GB | 1,523 GB |
| Non-torch (RCCL, HIP runtime) | 13.2 GB | 105 GB |
| CUDA-graph pool + safety | 6.6 GB | 53 GB |
| **KV cache pool** | **57.7 GB** | **462 GB** |

KV is **13,824 B/token/GPU**, which decodes exactly as
`(kv_lora 512 + qk_rope 64) × 1 B fp8 × 24 MLA layers`. Two consequences: **only the 24
full-attention layers consume paged KV** (the 69 KDA layers hold fixed-size recurrent state,
which is why a 93-layer model has a 24-layer KV footprint), and **KV is replicated across TP
ranks, not sharded** — MLA's latent is head-shared, so sharding would force a re-gather every
step. Capacity is 4.17 M tokens/GPU; even Run C's largest sweep used ~13% of it.

---

## 2. Results — all three runs

Output throughput (tok/s), ISL/OSL 1024/1024:

| Concurrency | A — original (cap 64) | B — MAD (cap 64) | **C — cap 256** |
|---:|---:|---:|---:|
| 1 | 46.1 | — | — |
| 8 | 288.0 | — | — |
| 32 | 824.0 | — | — |
| 64 | **1,258.5** | 1,142.7 | 1,237.0 |
| 128 | — | 1,187.3 | **1,792.5** |
| 256 | — | 1,182.4 | **2,482.2** |

Latency at the same points:

| Concurrency | B — MAD TTFT / TPOT (ms) | **C — cap 256 TTFT / TPOT (ms)** |
|---:|---:|---:|
| 64 | 281 / 52.97 | 283 / 49.98 |
| 128 | **49,456** / 53.35 | **346** / 70.98 |
| 256 | **149,723** / 53.85 | **463** / 103.72 |

### Three findings, in order of importance

**1. `max-num-seqs` was the binding limit — raising it 64 → 256 gave 2.1× throughput.**
Run C reaches **2,482 tok/s** where both 64-slot runs plateaued at ~1,180. At c=256 that is
**2.10×** Run B. Latency improved simultaneously: TTFT median fell **149,723 → 463 ms
(323×)**, because requests are now *served* rather than queued. TPOT rose 53.9 → 103.7 ms —
the honest cost of genuine batching, and a real trade rather than the pathological queueing
the 64-slot config produced.

**2. Sweeping concurrency past the cap measures queueing, not the engine.** Run B went
64 → 256 for **+3.5% throughput and 533× TTFT**, with TPOT flat at ~53 ms. Flat throughput,
flat TPOT, exploding TTFT is the signature of an admission cap: only the queue grows. This
matters for interpreting AMD's published MAD sweep, which uses concurrency 64/128/256 while
leaving `max-num-seqs` at 64 — those upper rows describe overload behaviour, not capability.

**3. AMD's MAD recipe was ~9% *slower*, not faster.** At the only matched point (c=64):
1,142.7 vs 1,258.5 tok/s = **0.91×**. Confidence is moderate, not high — single run each with
no repeats, and two variables moved together (image *and* launch config), so neither can be
isolated. What can be said safely: **there is no evidence here for adopting the MAD recipe**,
which is why Run C was built on Run A's configuration.

> **AMD's MAD image cannot serve Kimi-K3 as published.** It ships without
> `flash-linear-attention` (`fla`), which ATOM imports *unconditionally* for the KDA prefill
> path (`kimi_k3.py:749` — no flag guard, no fallback). 69 of 93 layers are KDA, so the server
> loads, answers `/v1/models`, then dies on the first real request with `ModuleNotFoundError`.
> Fixed by installing it in-container at startup. `rocm/atom-dev:latest` does ship it, so this
> is a regression in the dated tag. Full diagnosis in `notes-kimi-k3.md`.

---

## 3. What limits this workload

Utilization depends strongly on batch, so the columns below are labelled by concurrency, not
by run. **Comparing across different concurrencies is meaningless** — that is the trap
discussed in §4.

**Like-for-like, at matched c=64:**

| Resource | Run A (c=64) | Run C (c=64) | Capability |
|---|---:|---:|---|
| Compute | 1.1% | 1.1% | 2,500 TFLOP/s BF16 per GPU |
| **HBM bandwidth** | **28.6%** | **28.1%** | 8,000 GB/s per GPU |
| XGMI | 1.1% | 1.1% | ~537 GB/s per direction |

The two runs are **the same** at the same batch — raising `max-num-seqs` did not change
per-batch efficiency, it changed how large a batch the scheduler would admit.

**Run C at its own peak (c=256), which is a different operating point, not a regression:**

| Resource | Run C (c=256) | Capability |
|---|---:|---|
| Compute | 2.1% | 2,500 TFLOP/s BF16 per GPU |
| **HBM bandwidth** | **20.5%** | 8,000 GB/s per GPU |
| XGMI | 2.2% | ~537 GB/s per direction |

HBM% is *lower* at c=256 than at c=64 (20.5% vs 28.1%) even though throughput is **2× higher**
— because step time grows faster than weight traffic once attention and compute scale with
batch. §4 explains the mechanism and why this is expected rather than a problem.

**In every case HBM is the closest to saturation by a wide margin** — compute and interconnect
sit at ~1–2% throughout. To be precise about which bandwidth: this is **intra-GPU HBM**, each
GPU reading weights from its own on-package memory. It is *not* XGMI, which carries only
activation all-reduces. Traffic ratio is roughly **390:1** in HBM's favour.

### Why HBM traffic is so large: the MoE batching mechanism

A decode step must read every weight it activates. For MoE that depends on batch, because
tokens route independently:

| Batch | Experts fired/layer | Tokens per expert | Weights read/GPU/step |
|---:|---:|---:|---:|
| 1 | 18 | 1.0 | 3.4 GB |
| 64 | 610 | 1.7 | 116 GB |
| 128 | 805 | 2.5 | 153 GB |
| 256 | 887 | 4.6 | 169 GB |

`E × (1 − (1−1/E)^(B·topk))` — at batch 64, **610 of 896** experts fire, not 16. This is the
defining property of sparse MoE: **compute grows with batch, but weight traffic grows much
faster**, until nearly every expert is touched every step and traffic plateaus (~170 GB from
batch 512 on). MXFP4 is what makes it tractable — at BF16 the same reads would be 4× larger
and exceed HBM bandwidth outright.

### Communication is not a factor

With TP=8 and EP off, XGMI carries **only activation all-reduces**: 2 per layer × 93 layers =
**186 per token**, 14 KB per call, 2.67 MB/token. At Run C's peak that is a few GB/s against a
537 GB/s ceiling. Nothing else crosses the interconnect — weights are resident, KV is
replicated, gradients don't exist in inference, and there is no expert all-to-all.

One caveat against reading "~2% utilized" as "free": each call is 14 KB–896 KB, far below the
16 MiB–8 GiB range Part B's rccl-tests swept, so these collectives are **latency-dominated,
not bandwidth-dominated**. With 186 serialized calls per step, realistic per-call latency puts
their true cost at several percent of step time. Separately, this settles the Part B question:
the N=5/6/7 RCCL cliff is irrelevant here, since TP=8 is a power-of-2 arity that never
triggers it.

---

## 4. Next steps

### Can HBM utilization be pushed higher — and should it?

**A measured correction first.** `kimi-k3-base.md` §3.2 predicted HBM utilization would *rise* with
batch (~45% at 256, ~60% at 1024) as weight reads amortized. Run C shows it **falls**:

| Run | c=64 | c=128 | c=256 |
|---|---:|---:|---:|
| A — original (cap 64) | **28.6%** | — | — |
| B — MAD (cap 64) | 26.0% | 17.8% | 9.7% |
| **C — cap 256** | **28.1%** | 26.8% | **20.5%** |

The prediction failed because weight traffic plateaus as expected (116 → 169 GB, +45%) but
step time grows *faster* (51.7 → 103.1 ms, +99%), since attention, compute and collectives all
scale linearly with batch while weight reads do not. At c=256 the workload is no longer
weight-dominated — it is becoming balanced. (KV reads don't close the gap either: ~5.4 GB/step
moves 20.5% to only ~21%.)

Also note the headline "20% vs 29%" is **not** a regression: those are different
concurrencies. At matched c=64, Run C is 28.1% against Run A's 28.6% — the same.

**So: no, batching cannot raise HBM utilization, and it is the wrong target.** Throughput more
than doubled *while* HBM% fell. HBM% is a diagnostic of what limits you, not a goal — chasing
it means returning to c=64, which is 2× slower. If you specifically wanted higher HBM%, the
lever is reducing *non-weight* step time (attention kernels, the 186 serialized all-reduces,
scheduler overhead) so weight reads become a larger share — an ATOM/AITER kernel question,
not a configuration one.

### Ranked next experiments

1. **Extend Run C: `--max-num-seqs 512`, sweep to c=512.** Highest value. Run C's throughput
   was **still climbing at c=256 with no knee** (1,237 → 1,792 → 2,482), so the ceiling has
   not been found. Extend from **`kimi-k3-maxseqs.md`** — it has the best throughput and the
   better-performing image. Expect throughput to keep rising and HBM% to keep falling; that is
   the correct trade. KV headroom is ample (Run C used ~13% of the 57.7 GB pool). ~2 h.
2. **Profile a step.** Everything about *where* the non-weight ~80% of step time goes is
   inference from residuals, not measurement. A `rocprof` / torch-profiler trace would replace
   estimates with a real breakdown — and it is the prerequisite for judging any kernel-level
   idea, including EP (see below). Do this before optimizing anything kernel-side.
3. **ISL = 4096.** MAD's spec sweeps input lengths 1024 *and* 4096; all three runs here use
   1024 only. Longer prompts shift the prefill/decode balance and would likely change the
   high-concurrency picture.
4. **Repeats for the 9% MAD gap.** One run per config supports no claim about small
   differences being reproducible. Three repeats at c=64 would settle it.

### Should expert parallelism be retried on the MAD image?

**Recommendation: no — deprioritize it.** Not because it is impossible, but because the
*rationale* for EP has weakened since it was first proposed.

EP was motivated by §3's original reading: HBM at ~29%, XGMI at ~1%, so move traffic from the
loaded resource to the idle one. Run C undermines that. At c=256 HBM sits at **20.5%**, and
step time is increasingly dominated by attention, compute and scheduling rather than weight
reads. **EP would relieve something that is no longer the binding constraint** — best case a
few percent, and it cannot touch the ~80% of step time that is not weight traffic.

There is one genuine argument the other way, better than "it is a different build": MAD sets
`ATOM_USE_TRITON_MOE=0` and `AITER_USE_GROUPED_GEMM=0`, which **directly select a different
MoE kernel path** — and the failure came from the SiTUv2 MXFP4 kernel specifically. So it is
not implausible the EP-masking gap simply does not apply there. Against it: the MAD image is
*older* (dated 20260727) and measured ~9% slower overall, so even a working EP would build on
a worse baseline.

If closure is wanted anyway, it is cheap — the failure surfaces **at model load, ~10 minutes**,
not after a full sweep. But test it on a **freshly pulled `rocm/atom-dev:latest`** rather than
the MAD image: `:latest` has likely moved since 2026-08-14, and it is the faster baseline, so
a working EP there would be useful rather than academic.

### Known dead ends (tested)

- **Expert parallelism on `rocm/atom-dev:latest` (as of 2026-08-14).** ATOM raises
  `NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel
  masking` — MXFP4 experts and EP are mutually exclusive there. Tested on that image only;
  the MAD image was not pulled until four days later and EP was never retried on it, so this
  is image-specific evidence, not a universal claim. See the recommendation above before
  spending time on it.
- **The MAD recipe as a performance win.** Measured ~9% slower; adopt only if a future image
  changes that.

### Constraints to respect

- **Prefix caching must stay disabled.** KDA recurrent state is per-request and cannot be
  reconstructed from the paged MLA cache, so enabling it would be *silently incorrect*, not
  merely slower. In shared-prefix workloads this forfeits a win non-KDA models get free — an
  architectural trade, not an oversight.
- **The hybrid attention design is what makes 2.78 T fit at all.** Only 24 of 93 layers keep a
  growing KV cache. A conventional 93-layer model would need ~4× the KV per token.

---

## Source data

| What | Where |
|---|---|
| Run A (original) | `logs/atom/sweep_20260814_164903/` → `kimi-k3-base.md` |
| Run B (MAD) | `logs/atom/kimi_mad_20260818_223148/` → `kimi-k3-mad.md` |
| Run C (cap 256) | `logs/atom/kimi_maxseqs_20260819_171529/` → `kimi-k3-maxseqs.md` |
| A-vs-B detail | `kimi-k3-comparison.md` |
| MAD rationale, `fla` diagnosis | `notes-kimi-k3.md` |
| Cross-model context (Qwen3-8B, Llama-70B) | `atom.md` |

Derived figures (TFLOP/s, HBM %, expert activation, comms volumes) are computed from measured
throughput plus the parsed architecture; the memory table is read verbatim from the server's
own budget line. The KDA parameter count is a config-derived approximation (±15% on the 84 B
active figure) — the "~1–2% of compute peak" conclusion has far too much margin to be affected.
