# AMD Cloud MI355X — cross-suite summary

Benchmarking 8 × AMD Instinct MI355X (gfx950), ROCm 7.14, Ubuntu 22.04.5, Docker.
Four suites, run strictly sequentially (each wants all 8 GPUs and the full ~11.2 kW tray).

| Part | Suite | Status | Detail |
|---|---|---|---|
| A | ROCm Validation Suite — `gst` TFLOPS + health | ✅ complete | `rvs_tflops.md`, `fp4_investigation.md` |
| B | rccl-tests — collective bandwidth | ✅ complete | `rccl.md`, `rccl_busbw.png` |
| C | Primus — GEMM/attention/RCCL + Megatron-LM | ✅ complete | `PRIMUS_REPORT.md` |
| D | ATOM — LLM inference serving | ✅ complete (11 Kimi-K3 experiments) | `atom.md`, `kimi-k3-improve.md` |

The methodological premise throughout: **`dell-cloud/` is the reference**, this host is the
reproduction. Where numbers differ, the interesting question is *why*, since both machines
are 8 × MI355X and only the software stack differs (ROCm 7.2.3 + gfx942 alias there,
ROCm 7.14 native gfx950 here).

---

## 1. Headline numbers

| Measurement | Result | Reference |
|---|---:|---|
| Peak GEMM (RVS `gst`, bf16, N=1) | **1,628 TF/s/GPU** | 65% of 2,500 dense peak |
| HBM bandwidth (RVS `babel`, streaming read) | **7,260 GB/s** | 91% of 8 TB/s spec |
| RCCL all-reduce, N=8 | **396.6 GB/s** busbw | 74% of 537.6 GB/s per-direction |
| Megatron llama2-7B (Primus, N=8, compute) | **1,135 TF/s/GPU** | 87% of `gemm-dense` ceiling |
| LLM serving, Llama-70B FP8 TP=8 | **9,342 tok/s** | c=256 |
| LLM serving, Kimi-K3 2.78 T TP=8 | **3,386 tok/s** | c=512, `max-num-seqs 512` |
| Kimi-K3 single-stream (one user) | **46.6 tok/s** | c=1; 6.5 tok/s at c=512 |

---

## 2. The five findings that matter

### 2.1 fp4 gains 26% from native gfx950 — everything else is unchanged

Same silicon as Dell Cloud, different stack. Eight of nine precisions land within **1%** of
the Dell baseline; **fp4 alone gains 1.26×**. That selectivity is the finding: it points at
the gfx942 alias specifically penalising the MX-FP4 kernel path, and it is the concrete
payoff for not setting `HSA_OVERRIDE_GFX_VERSION` on this host.

**But fp4/fp6/bf6 sit far below peak** — 40% / 12% / 12%, against 71% for fp8 and 97% for
fp32/fp64. Three separable causes, established by measurement rather than assumption:
memory bandwidth is *not* one of them (these GEMMs use 1–8% of HBM). MX block scaling costs
throughput; **fp6 is not byte-aligned** (4 values per 3 bytes) and is *absolutely slower than
fp8* despite twice the nominal peak — the signature of a non-native path; and native codegen
improved fp4 by 26% while moving fp6 not at all, separating a tuning problem from a
structural one. → `rvs_tflops.md`

### 2.2 The non-power-of-2 RCCL cliff reproduces, and is ~20% shallower

Every ring-based collective loses **67–81%** of its bandwidth at N=5/6/7. all_reduce drops
169.5 → 48.0 GB/s from N=4 to N=5. This reproduces dell-cloud's finding on a newer stack.

The newer RCCL (2.30.4 vs 2.27.7) helps **only where the ring breaks**: non-power-of-2
arities average **1.25×** the Dell numbers while power-of-2 arities sit at **0.97×** — parity.
That asymmetry is itself the evidence: a kernel-level speedup would lift every N, so the gain
must come from *ring-construction logic*, not codegen. Root-cause analysis lives in
dell-cloud's `summary-power2.md`; this run reproduces and quantifies it rather than
re-deriving it. → `rccl.md`

### 2.3 Megatron realizes 87% of the achievable GEMM rate

A ladder of ceilings, each gap attributing a specific loss:

| Ceiling | TF/s/GPU | Megatron as % |
|---|---:|---:|
| RVS `gst` bf16 (silicon, no framework) | 1,639 | 69% |
| Primus `gemm` (square 4096³) | 1,444 | 79% |
| Primus `gemm-dense` (dense-model shapes) | 1,309 | **87%** |
| Megatron llama2-7B | 1,135 | 100% |

87% against a realistic shape mix is a good result. The residual is non-GEMM work, and
attention is the leading candidate — it runs at **55% (fwd) / 17% (bwd)** of the GEMM rate.

> This ladder is the **Primus** llama2-7B path, all measured on this host. It is unrelated to
> the `megatron-ref` GPT-15.6B reproduction in `PRIMUS_REPORT.md` §1.2, which never ran
> successfully here — see §5.

Two traps documented here because both produced wrong conclusions before being caught: the
Megatron log emits **two** TFLOP/s figures (`compute per GPU` ≈ 1,135 vs `throughput per GPU`
≈ 294, ~4× apart), and comparing across them manufactures a spurious regression; and Megatron
here is **pure data parallel** (DP=N, TP=PP=1), which is why it is largely insensitive to the
RCCL cliff — one gradient all-reduce per ~5 s iteration is negligible even at degraded
bandwidth. → `PRIMUS_REPORT.md`

### 2.4 A 2.78 T model serves on one node — and the limit was a config flag

Kimi-K3 (2.78 T params, 1.5 TB) runs at TP=8 with **1.5 TB of weights across 2.3 TB of HBM**.

The most actionable result in the whole project: throughput was capped at ~1,180 tok/s by
**`--max-num-seqs 64`**, not by hardware. Raising it to 256 gave 2,482 tok/s, and to 512 gave
**3,386 tok/s (2.9×)** while TTFT median *improved* 149,723 → 541 ms (**277×**), because
requests were being served instead of queued. Nothing was saturated — HBM ~29%, compute ~1%,
XGMI ~1%.

**The ceiling was then found: 1024 is past the knee.** At `max-num-seqs 1024` throughput
*falls* to 1,946 tok/s at c=512 (−43% against cap 512) with TTFT at 366 s. **512 is the
setting**; the curve is not monotonic and cannot be extrapolated.

This also means AMD's published MAD sweep (concurrency 64/128/256 with `max-num-seqs=64`)
characterises **queueing behaviour**, not serving capability, above 64. → `kimi-k3-improve.md`

### 2.5 The bottleneck is latency, not bandwidth — and configuration cannot reach it

At every operating point **no resource is above ~30%**: compute ~1–2%, XGMI ~1–2%, HBM
~20–28%. When nothing is near its ceiling and throughput is still limited, the limiter is
serialization, not bandwidth. Two supporting facts: the 28% HBM figure is measured against a
*sequential streaming* ceiling that gather-style MoE expert reads cannot reach, and from
c=64→256 weight traffic grows +45% while step time grows +99%.

At c=1 a step reads ~3.4 GB of weights ≈ 1.5 ms of traffic against a measured 21.5 ms TPOT,
so **~93% of a single-request step is not weight reading**. A CPU-side trace re-analysis
(8 ranks, agreeing within ±0.5 pp) locates that 93%:

| Bucket | Share | | Bucket | Share |
|---|---:|---|---|---:|
| tensor copy/reshape | **41.3%** | | GEMM | 4.9% |
| KDA linear attention | **38.6%** | | MoE routing/experts | 1.3% |
| MLA full attention | 6.6% | | **collectives** | **0.8%** |

**The 186 all-reduces are not the problem** — 0.8%, the smallest bucket. Tail latency agrees
independently: TPOT p99/median stays at 1.00–1.11 from c=1 to c=512, so decode is metronomic,
not barrier-stalled. What dominates is the **linear-attention path and the tensor copies
around it (~80% together)** — `aten::copy_` runs 23,447 times per capture, ~21 per KDA call,
which looks like layout conversion in the `fla` integration rather than required work.

**That headroom is real but unreachable from configuration** — a kernel-path sweep at low
batch moved single-stream speed by **<5%** (§3), and TP=8 is forced by 1.5 TB of weights, the
186 collectives have no flag, and `num_nextn_predict_layers = 0` rules out speculative
decoding. Further gains require ATOM/AITER kernel work. → `kimi-k3-single-stream.md`

---

## 3. What was wrong, and got corrected

Recorded because each was believed at some point and would have misled:

| Claim | Correction |
|---|---|
| ROCm installed runtime-only (plan §0) | Dev packages were already present; no `apt` needed |
| fp6/bf6 peak = 5,000 TFLOPS | **10,000** — CDNA 4 runs MX-FP6 at FP4 rate; inflated "% of peak" 2× |
| RVS `pqt` module measures P2P XGMI | **No `pqt` module exists**; it is `pbqt`. As written, the health script would have skipped the one test Part B's fabric-vs-algorithm attribution rests on |
| Megatron config `llama2_7B-pretrain.yaml` | Per-arch now: `MI355X/llama2_7B-BF16-pretrain.yaml` |
| Primus 294 TF/s vs reference 790 = regression | Different metrics (wall-clock vs compute); on the same metric it is 1,135 vs 1,132 — parity |
| MAD recipe should be faster | Measured **~9% slower** at matched concurrency (repeat test running to confirm) |
| HBM % should rise with batch | It **falls** (28% → 20%) while throughput doubles; step time grows faster than weight traffic |
| EP is the top tuning lever | **EP works on the MAD image** (the `NotImplementedError` is specific to the SiTUv2 kernel on `:latest`, confirmed still failing on a freshly pulled tag) — but at matched `max-num-seqs` it **loses at every batch**: 0.85× / 0.94× / 0.94× at c=64/128/256 |
| EP gains 1.38× at c=128 | Confounded — that A/B compared `max-num-seqs 64` against 256, so it measured the admission cap, not EP. Only matched-cap runs are valid |
| MAD is ~9% slower (n=1) | Confirmed and **larger: 0.864×, i.e. ~14% slower** (n=3 per config, within-config spread ~2–4%) |
| HBM at 28% means 72% headroom | The denominator is *streaming* peak; scattered MoE expert gathers cannot reach it. Latency, not bandwidth, is the limiter |
| The 186 all-reduces dominate step time | **0.8% measured** — the smallest bucket. "~1% XGMI" means the collectives are cheap, not that they are latency-bound. Tensor copies (41.3%) + KDA attention (38.6%) are ~80% of host step time |
| `megatron-ref` reproduces Dell | The completed run was **not** Dell-comparable (8 flags missing → different model, ~3× slower). With them added it SIGSEGVs on the first step; **still unmeasured** |

Two infrastructure bugs are worth carrying forward: rccl-tests binaries ship **without
RPATH**, so `LD_LIBRARY_PATH=/opt/rocm/lib` is required or every collective fails at exec;
and AMD's MAD-pinned Kimi-K3 image is **missing `flash-linear-attention`**, which ATOM imports
unconditionally for the KDA path — so that image cannot serve Kimi-K3 without patching.

---

## 4. Cross-suite reading

The suites form a ladder, and each rung licenses a conclusion about the one above:

1. **RVS health** (`pbqt` XGMI, `pebb` PCIe, `babel` HBM) — is the fabric sound? A clean
   result is what lets a later RCCL cliff be blamed on the *algorithm* rather than hardware.
2. **RVS `gst`** — the compute ceiling per precision.
3. **rccl-tests** — the collective ceiling at the RCCL API.
4. **Primus microbenches** — what PyTorch gets from 2 and 3.
5. **Megatron / ATOM** — what a real model gets from all of the above.

Applied: the RCCL cliff at N=5/6/7 (rung 3) is **irrelevant to both real workloads** — Megatron
because DP issues one all-reduce per ~5 s iteration, ATOM because TP=8 is a power-of-2 arity
that never triggers it. A fabric defect that never reaches production configurations.

One caveat that recurs at every rung: **bandwidth utilization measures throughput headroom,
not latency exposure.** ATOM's 186 all-reduces per token move trivial bytes (~1% of XGMI) but
are latency-dominated at 14 KB–896 KB per call — far below the 16 MiB–8 GiB range Part B swept.
"1% utilized" is a floor on cost, not an estimate of it.

---

## 5. Status and what remains

**All GPU work on this host is complete.** Parts A–D, the fp4 investigation, the RCCL config
sweep, and **eleven** Kimi-K3 experiments. The machine went idle 2026-08-20 19:10 UTC.

**Kimi-K3 experiment ledger** — every run and what it settled:

| Experiment | Result |
|---|---|
| Base run (`:latest`, cap 64) | 1,258 tok/s peak; 46.6 tok/s single-stream at c=1 |
| MAD recipe (cap 64) | 1,187 tok/s — slower, not faster |
| `max-num-seqs` 256 | 2,482 tok/s (2.0×) |
| `max-num-seqs` 512 | **3,386 tok/s (2.9×) — the peak** |
| `max-num-seqs` 1024 | 1,946 tok/s — **past the knee**, 512 is the setting |
| ISL = 4096 | 2,098 tok/s at c=256; prefill length does not change the picture |
| Repeats × 3 per config | MAD gap real: **0.864×**, spread ~2–4% within config |
| EP on `:latest` (fresh pull) | Still `NotImplementedError` — SiTUv2 kernel limitation |
| EP on MAD image | **Works** — corrects the "mutually exclusive" claim |
| EP vs TP, matched cap | **EP loses at every batch**: 0.85× / 0.94× / 0.94× |
| Single-stream kernel sweep | **Null (<5%)** — config-level tuning for TPOT is closed |

**Known gaps, honestly stated:**

- **`megatron-ref` never produced a number.** Four attempts. The one that *completed*
  (2026-08-20 07:45) was later found **not Dell-comparable** — eight flags missing, so it
  built a different model and ran ~3× slower than Dell on a *cheaper* one. With the flags
  corrected the model matches Dell exactly (16,223,016,960 params) but SIGSEGVs on the first
  training step; dropping the overlap flags, then gradient-accumulation-fusion, did not help.
  **`PRIMUS_REPORT.md` §1.2 therefore carries only external reference data** — neither the
  B200 nor the Dell MI355X figure was measured on this host.
- **The profiler traces contain no GPU kernel events**, only `cpu_op`. Re-analysed
  (`kimi-k3-profile.md`) to get an exact exclusive-time breakdown of *host* step time, which
  is what corrected the collectives ranking. But it is **not** a kineto category-name mismatch
  as first supposed — GPU capture was never enabled, so **kernel-level timing is
  unrecoverable from these files** and the breakdown is host-side and directional. A future
  profiling run must enable GPU activity explicitly; that is the one measurement still
  genuinely missing.
- **`K2_triton_moe` did not load** in the single-stream sweep (`ATOM_USE_TRITON_MOE=1` died
  during model load), so that arm is missing. Three of four arms completed and agree within
  ±3%, so the null result stands, but the Triton MoE path is untested at low batch.
- ~~Tail latency unexamined.~~ **Done** — `kimi-k3-profile.md`. TPOT p99/median 1.00–1.11
  across the whole range (decode is steady); TTFT p99/median 1.2 → 49 (queueing at admission).
- **Part A leftovers**: `iet` and `rvs_level_5` hit timeouts (rc=124); `rvs_level_4`
  core-dumped. Not retried.
- **Small n almost everywhere.** Only the MAD gap was repeated (n=3).

## 6. Files

| File | Contents |
|---|---|
| `rvs_tflops.{md,csv}` | Part A — TFLOPS × 9 precisions × N=1..8, Dell/AMD/B200 comparison |
| `fp4_investigation.md` | Part A — fp4 N≥5 scaling anomaly (non-deterministic) |
| `rccl.{md,csv}`, `rccl_busbw.png` | Part B — collective bandwidth, cliff analysis |
| `PRIMUS_REPORT.md` | Part C — GEMM/attention/RCCL microbenches + Megatron |
| `atom.{md,csv}` | Part D — three model tiers at a glance |
| `kimi-k3-improve.md` | **Kimi-K3 entry point** — all runs consolidated + next steps |
| `kimi-k3-base.md`, `-mad.md`, `-maxseqs.md`, `-maxseqs512.md`, `-maxseqs1024.md`, `-comparison.md` | Per-run Kimi-K3 detail |
| `kimi-k3-isl4096.md` | ISL = 4096 sweep |
| `kimi-k3-repeats.md` | Repeatability of the MAD gap (n=3 per config) |
| `kimi-k3-ep-matched.md` | EP vs TP-only at matched `max-num-seqs` — the valid A/B |
| `kimi-k3-single-stream.md` | Per-request speed vs kernel path (null result) |
| `kimi-k3-profile.md` | Profiler run — **no usable kernel data**, see gaps above |
| `../notes.md`, `../notes-kimi-k3.md` | Host layout; Kimi-K3 rerun rationale and diagnoses |
| `../plan.md` | The benchmark plan, with a setup log of what actually happened |
