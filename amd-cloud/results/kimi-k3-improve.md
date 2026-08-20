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

### Per-request speed — what one user actually experiences

The tables above are **aggregate** throughput, summed over all concurrent requests. The number
a single user sees is `1000 / TPOT` — the decode rate of *one* stream. It moves in the
**opposite direction**, and it is the metric no experiment here has yet improved:

| Concurrency | A — original | B — MAD | C — cap 256 | D — cap 512 |
|---:|---:|---:|---:|---:|
| **1** | **46.6** | — | — | — |
| 2 | 44.3 | — | — | — |
| 4 | 40.0 | — | — | — |
| 8 | 37.0 | — | — | — |
| 16 | 32.0 | — | — | — |
| 32 | 26.4 | — | — | — |
| 64 | 20.0 | 18.9 | 20.0 | 20.0 |
| 128 | — | 18.7 | 14.1 | 14.1 |
| 256 | — | 18.6 | 9.6 | 9.8 |
| 512 | — | — | — | **6.5** |

*Per-request tok/s = `1000 / median TPOT`. Sources as in §Source data.*

**The peak aggregate result and the peak per-request result are at opposite ends of the
table.** Run D's headline 3,385.9 tok/s at c=512 is simultaneously the **worst** single-stream
experience measured — 6.5 tok/s per user, 7× slower than c=1. Batching does not create
throughput out of nothing; it trades latency for it.

**Two readings that are easy to get wrong here:**

- **B's flat column is an artifact, not stability.** B holds ~18.6–18.9 tok/s across c=64→256
  only because `max-num-seqs 64` capped the *running* batch at 64 regardless of requested
  concurrency. Its TPOT stayed flat because the effective batch stayed flat; the queue absorbed
  everything else, which is why TTFT went to 149,723 ms. Read B's column as "per-request speed
  at effective batch 64", measured three times.
- **C and D agree closely** at every shared point (20.0/14.1/9.6 vs 20.0/14.1/9.8), which is
  the useful confirmation: per-request speed is a function of *effective batch size*, not of
  the cap setting. Raising the cap does not slow anyone down — admitting more concurrent work
  does.

**c=1 was measured once, in Run A only**, at **46.6 tok/s (TPOT 21.48 ms)**. Every subsequent
run starts at c=64. So the single-stream floor rests on one measurement from the original
image and has never been reproduced, nor tested against any other kernel configuration — see
§4 *Improving per-request speed*.

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

**HBM is the closest to saturation by a wide margin** — compute and interconnect sit at
~1–2% throughout. To be precise about which bandwidth: this is **intra-GPU HBM**, each GPU
reading weights from its own on-package memory. It is *not* XGMI, which carries only
activation all-reduces. Traffic ratio is roughly **390:1** in HBM's favour.

### The bottleneck is latency, not bandwidth

Closest to saturation is not the same as saturated, and this is the single most important
thing to take from the table above. **No resource is above ~30%.** Compute ~1–2%, XGMI ~1–2%,
HBM ~20–28%. When nothing is near its ceiling and throughput is still limited, the limiter is
not any bandwidth — it is **latency and serialization**: time spent waiting on dependent
operations rather than time spent moving bytes.

This reframes the whole file. "HBM is the bottleneck" is the right answer to *which resource
is under the most pressure*, and the wrong answer to *what should be fixed*. Two reasons:

**1. The 28% is measured against a ceiling this access pattern cannot reach.** The 8,000 GB/s
figure — and the 7,260 GB/s Part A `babel` measurement, 91% of it — is **sequential streaming
read**. Kimi's expert reads are the opposite of streaming: top-16 of 896 experts, each GPU
holding a 1/8 TP slice, gathered from scattered locations, with MXFP4 block-scale dequant on
top. Gather-style reads of many medium-sized blocks never approach streaming peak. So 28% of
*streaming* peak is not 72% of headroom — it may already be close to the practical ceiling
**for this pattern**, and 90% was never reachable. Quoting HBM% against the spec number
overstates how much is being left on the table.

**2. The step-time arithmetic points away from weight traffic.** From c=64 to c=256, weight
traffic grows **+45%** (116 → 169 GB) but step time grows **+99%** (51.7 → 103.1 ms). The
extra time is going somewhere that scales with batch while weight reads plateau — attention,
expert routing, and collectives.

**The serialization suspects, in order of suspicion:**

- **186 all-reduces per token** (2 × 93 layers) at 14 KB–896 KB each. These are far below the
  size where bandwidth matters, so each costs a round-trip plus a synchronization barrier.
  The "~1% XGMI" is the **signature** of a latency-bound collective, not evidence of spare
  capacity — see *Communication is not a bandwidth factor* below, which is precise about
  that distinction.
- **69 of 93 layers are KDA** linear attention, carrying recurrent state with a sequential
  dependency from one step to the next and low arithmetic intensity.
- **Kernel launch and scheduler overhead**, which becomes a meaningful fraction when
  per-kernel work is small.

**This ranking is inference, not measurement — and the run that was supposed to settle it
failed.** The profiling step (next-step #2) captured 8 traces but the analyzer found no GPU
kernel events in them (702,510 events parsed, none carrying a GPU kernel category — most
likely a kineto category-name mismatch, not an empty trace). See `kimi-k3-profile.md`. The
95 MB of raw traces are retained at `/mnt/scratch/shaohao/traces/kimi_20260820_041644`, so
re-parsing them costs **no GPU time**. Until that is done, the split between collectives, KDA
and launch overhead is unresolved, and no kernel-level optimization should be chosen on the
strength of the ordering above.

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

### Communication is not a *bandwidth* factor

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

<!-- BEGIN run-d-maxseqs512 (auto-generated) -->

### Run D — `max-num-seqs 512` (next-step #1, completed)

Ran the top-ranked follow-up: raise the cap again and sweep further. Source: `logs/atom/kimi_512_20260819_211857/`, detail in `kimi-k3-maxseqs512.md`.

| Concurrency | Run C (cap 256) | **Run D (cap 512)** | D / C | D TTFT med (ms) | D TPOT med (ms) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1,237.0 | 1,236.0 | **1.00×** | 285.2 | 50.08 |
| 128 | 1,792.5 | 1,795.5 | **1.00×** | 387.2 | 70.71 |
| 256 | 2,482.2 | 2,529.9 | **1.02×** | 458.3 | 101.52 |
| 512 | — | **3,385.9** | — | 541.2 | 152.82 |

**Still climbing.** c=256 → 512 gained **34%** (2,529.9 → 3,385.9 tok/s). The admission cap was still the binding limit at 256; the ceiling has *still* not been found. Raising it further is worth another round, though TPOT (152.82 ms) is now the thing to watch — at some point the latency cost stops being acceptable even if throughput rises.

At the peak point (c=512): **896 of 896 experts** fire per layer, 170.5 GB/GPU/step of weight traffic, step 151.2 ms → HBM **~14%**. As predicted in §4, HBM utilization keeps falling as batch grows while throughput rises — the two move in opposite directions, which is why HBM% is a diagnostic and not a target.

<!-- END run-d-maxseqs512 -->

<!-- BEGIN next-steps-3-4 (auto-generated) -->

### ISL = 4096 (next-step #3, completed)

Same config as Run C (cap 256) with **input length 4096 instead of 1024** — ISL is the only variable. Source: `logs/atom/kimi_isl4096_20260820_131217/`, detail in `kimi-k3-isl4096.md`.

| Concurrency | ISL 1024 (Run C) | **ISL 4096** | 4096/1024 | TTFT med (ms) | TPOT med (ms) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1,237.0 | **1,225.3** | **0.99×** | 318.6 | 50.08 |
| 128 | 1,792.5 | **1,671.0** | **0.93×** | 471.9 | 75.01 |
| 256 | 2,482.2 | **2,098.2** | **0.85×** | 531.6 | 120.03 |

**4× longer prompts cost 15% throughput** at c=256. Prefill work scales with ISL and competes with decode for the same GPU, so a bigger share of each step goes to prompt processing. Expected direction; the magnitude is the useful number for capacity planning.

### Repeatability of the MAD gap (next-step #4, completed)

Three repeats at c=64 per config, to test whether the single-run **0.91×** figure in `kimi-k3-comparison.md` is real. Source: `logs/atom/kimi_repeats_20260820_135851/`, detail in `kimi-k3-repeats.md`.

| Config | n | mean tok/s | stdev | rel. spread |
|---|---:|---:|---:|---:|
| `A_original` | 3 | **1,365.2** | 15.6 | 2.1% |

> Repeats share one server process per config, so this bounds benchmark-to-benchmark variance, not full cold-start variance.

<!-- END next-steps-3-4 -->

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
it means returning to c=64, which is 2× slower.

**And 28% is not as low as it looks.** It is measured against 8,000 GB/s, which is the
*sequential streaming* rate. Reading top-16-of-896 experts as scattered 1/8 TP slices with
MXFP4 dequant is a gather, not a stream, and gathers do not reach streaming peak — so a large
part of the apparent 72% gap is unreachable by construction rather than lost to inefficiency.
Pushing toward 90% is not a realistic target for this access pattern at any batch size.

**The real reason it cannot be pushed higher is that HBM is not what is holding the workload
back.** Nothing is above ~30% — compute ~1–2%, XGMI ~1–2%, HBM ~20–28% — which means the
binding constraint is **latency and serialization, not bandwidth** (§3, *The bottleneck is
latency, not bandwidth*). Raising HBM% would require shrinking *non-weight* step time — the
186 serialized all-reduces, the 69 KDA layers' sequential state updates, kernel launch
overhead — so that weight reads become a larger share of a shorter step. That is an
ATOM/AITER kernel question, not a configuration one, and the profiling run meant to identify
which of those dominates did not produce usable kernel data (see §3). Re-parsing the retained
traces is the cheapest next move, and it costs no GPU time.

### Improving per-request speed (single-stream tok/s)

**No experiment run so far has targeted this metric.** Every Kimi-K3 experiment to date —
caps 64/256/512/1024, ISL=4096, MAD recipe, EP, repeats — measured *aggregate* throughput, and
each either left per-request speed unchanged (~20 tok/s at c=64 in A, C and D alike) or made
it worse by admitting a larger batch. The single-stream number has exactly **one** measurement
behind it: **46.6 tok/s at c=1** in Run A.

#### Is it improvable? Yes in principle, no by configuration

Two questions get confused here, and they have different answers:

| Question | Answer |
|---|---|
| Is there headroom in single-stream speed? | **Yes** — and it is large |
| Can any setting on this server realize it? | **No** — every config lever is tested or closed |
| What would realize it? | **ATOM/AITER kernel changes** — code, not configuration |

**The headroom is arithmetic, not opinion.** At c=1 a decode step reads ~3.4 GB of weights per
GPU. At the ~2.2 TB/s effective rate the c=64 measurement implies (116 GB / 51.7 ms), those
reads should take **~1.5 ms**. Measured TPOT at c=1 is **21.48 ms**. So **~93% of a
single-request step is not weight reading** — it is the serialization described in §3, and
unlike a bandwidth wall, serialization is compressible. That gap is a property of the workload
and holds regardless of what any experiment returns.

**But nothing in a config file reaches it.** TP=8 is forced by 1.5 TB of weights against
2.3 TB of HBM; the 186 all-reduces are fixed by TP × 93 layers with no flag to reduce them;
`num_nextn_predict_layers = 0`, so there are no MTP heads for speculative decoding; HIP graphs
are already enabled. Kernel-path selection is the **only** untested configuration knob, which
is precisely what the experiment below sweeps — and why a small or null result there is the
expected outcome rather than a surprise.

**So the honest form of the answer is: yes, but it needs engine work, not tuning.** Fusing or
batching the per-layer collectives, or breaking the sequential dependency across the 69 KDA
layers, is a change to ATOM/AITER. The experiment below does not test that and cannot; what it
does is **close out the configuration question**, so that any remaining effort goes to the
kernels instead of to more flag sweeps.

**What could move it, and what could not:**

| Lever | Available here? | Why |
|---|---|---|
| Kernel path selection (Triton vs AITER for GEMM / MoE / attention) | **Yes — untested at low batch** | At c=1 kernel efficiency and launch overhead dominate; every kernel comparison so far was run at c=64+, where batching hides exactly the costs that matter at c=1 |
| Reducing collective count | No | 186 all-reduces is fixed by TP=8 × 93 layers; there is no flag |
| TP < 8 with replicas | No | 1.5 TB of weights against 2.3 TB of HBM forces TP=8 on one node |
| Speculative decoding / MTP | No | `num_nextn_predict_layers = 0` — Kimi-K3 ships no MTP heads, so the standard single-stream multiplier is unavailable without an external draft model |
| HIP graphs | Already on | Server log reports `cudagraph=True`; launch overhead is already partly mitigated |

Only the first row is testable as a configuration change, which makes it the whole experiment.

**Set up (not yet run): `atom/run_single_stream.sh`.** Sweeps the **low-concurrency** regime
(c=1/2/4/8) across four kernel configurations on the MAD image, reporting TPOT and per-request
tok/s rather than aggregate throughput. Configurations flip one kernel-path decision at a time
from the MAD baseline:

| Config | Change from MAD baseline | Isolates |
|---|---|---|
| `K1_mad_default` | none (reference) | — |
| `K2_triton_moe` | `ATOM_USE_TRITON_MOE=1` | MoE kernel path |
| `K3_aiter_attn` | `ATOM_USE_UNIFIED_ATTN=0`, `ATOM_FORCE_ATTN_TRITON=0` | attention path (the 69 KDA + 24 MLA layers) |
| `K4_grouped_gemm` | `ATOM_USE_TRITON_GEMM=0`, `AITER_USE_GROUPED_GEMM=1` | GEMM / grouped-GEMM path |

One variable per arm, so a difference is attributable. Run A's c=1 figure of 46.6 tok/s is the
number to beat, with the caveat that it was measured on `rocm/atom-dev:latest` and these arms
run on the MAD image — `K1_mad_default` therefore doubles as the matched control that makes
the comparison legitimate. ~1.5 h for all four.

**Expectations, stated in advance:** effects should be modest — single-digit to low-tens of
percent. The MAD kernel set was already measured ~9% *slower* in aggregate at c=64, which is
evidence these knobs matter at the margin rather than the order of magnitude. **A negative
result is still worth having**, because it would localize the 93% to launch overhead and the
KDA dependency chain — things no environment variable can reach — and would close
configuration-level tuning for this model.

**Do the trace re-parse first if possible.** It costs no GPU time and would say whether
attention, collectives or launch overhead dominates at low batch, which turns this sweep from
four guesses into a targeted test. See §3.

### Ranked next experiments

1. ~~**Extend Run C: `--max-num-seqs 512`, sweep to c=512.**~~ **DONE — see Run D above.** Was highest value. Run C's throughput
   was **still climbing at c=256 with no knee** (1,237 → 1,792 → 2,482), so the ceiling has
   not been found. Extend from **`kimi-k3-maxseqs.md`** — it has the best throughput and the
   better-performing image. Expect throughput to keep rising and HBM% to keep falling; that is
   the correct trade. KV headroom is ample (Run C used ~13% of the 57.7 GB pool). ~2 h.
2. **Profile a step.** **ATTEMPTED — traces captured, analysis FAILED.** Still the highest-value
   open item. Everything about *where* the non-weight ~80% of step time goes is inference from
   residuals, not measurement, and it is the prerequisite for judging any kernel-level idea,
   including EP (see below). The run captured 8 traces (95 MB, retained at
   `/mnt/scratch/shaohao/traces/kimi_20260820_041644`) but `analyze_profile.py` found no GPU
   kernel events among 702,510 parsed — a kineto category-name mismatch, most likely. **Fixing
   the parser needs no GPU time**, so this should be done regardless of machine availability.
   ATOM ships `tools/analyze_trace_summary.py`, which is the obvious thing to try first.
3. ~~**ISL = 4096.**~~ **DONE — see above.** MAD's spec sweeps input lengths 1024 *and* 4096; all three runs here use
   1024 only. Longer prompts shift the prefill/decode balance and would likely change the
   high-concurrency picture.
4. ~~**Repeats for the 9% MAD gap.**~~ **DONE — see above.** One run per config supports no claim about small
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
