# AMD Cloud MI355X — cross-suite summary

Benchmarking 8 × AMD Instinct MI355X (gfx950), ROCm 7.14, Ubuntu 22.04.5, Docker.
Four suites, run strictly sequentially (each wants all 8 GPUs and the full ~11.2 kW tray).

| Part | Suite | Status | Detail |
|---|---|---|---|
| A | ROCm Validation Suite — `gst` TFLOPS + health | ✅ complete | `rvs_tflops.md`, `fp4_investigation.md` |
| B | rccl-tests — collective bandwidth | ✅ complete | `rccl.md`, `rccl_busbw.png` |
| C | Primus — GEMM/attention/RCCL + Megatron-LM | ✅ complete | `PRIMUS_REPORT.md` |
| D | ATOM — LLM inference serving | ✅ complete (+ follow-ups running) | `atom.md`, `kimi-k3-improve.md` |

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
| LLM serving, Kimi-K3 2.78 T TP=8 | **2,482 tok/s** | after raising `max-num-seqs` |

---

## 2. The four findings that matter

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

Two traps documented here because both produced wrong conclusions before being caught: the
Megatron log emits **two** TFLOP/s figures (`compute per GPU` ≈ 1,135 vs `throughput per GPU`
≈ 294, ~4× apart), and comparing across them manufactures a spurious regression; and Megatron
here is **pure data parallel** (DP=N, TP=PP=1), which is why it is largely insensitive to the
RCCL cliff — one gradient all-reduce per ~5 s iteration is negligible even at degraded
bandwidth. → `PRIMUS_REPORT.md`

### 2.4 A 2.78 T model serves on one node — and the limit was a config flag

Kimi-K3 (2.78 T params, 1.5 TB) runs at TP=8 with **1.5 TB of weights across 2.3 TB of HBM**.

The most actionable result in the whole project: throughput was capped at ~1,180 tok/s by
**`--max-num-seqs 64`**, not by hardware. Raising it to 256 gave **2,482 tok/s (2.1×)** while
TTFT median *improved* 149,723 → 463 ms (**323×**), because requests were being served
instead of queued. Nothing was saturated — HBM ~29%, compute ~1%, XGMI ~1%.

This also means AMD's published MAD sweep (concurrency 64/128/256 with `max-num-seqs=64`)
characterises **queueing behaviour**, not serving capability, above 64. → `kimi-k3-improve.md`

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
| EP is the top tuning lever | ATOM raises `NotImplementedError` — MXFP4 experts and EP are mutually exclusive on the tested image |

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

**Complete and reported**: Parts A–D, plus the fp4 scaling investigation, the RCCL config
sweep, the Kimi-K3 deep-dive, and three Kimi-K3 configuration experiments.

**Running unattended at time of writing** (queued sequentially, auto-updating their reports):

| Job | Answers |
|---|---|
| `max-num-seqs 512` | Has the throughput ceiling been found? |
| ISL = 4096 | How does 4× prompt length shift prefill/decode? |
| Repeats × 3 | Is the "MAD is 9% slower" gap real or noise? |
| `megatron-ref` | Places this host in the same table as B200 / Dell MI355X |
| Bandwidth health recheck | Clean `pebb`/`pbqt` — the first pass ran during a 1.5 TB download |
| Profiler trace | Replaces the residual-based step-time estimate with measurement |

**Known gaps, honestly stated:**

- **`megatron-ref` has never completed.** It failed twice (hipBLASLt algorithm selection, then
  RCCL requiring `HSA_NO_SCRATCH_RECLAIM=1`); both fixes are applied and it is queued, but
  until it finishes, `PRIMUS_REPORT.md` §1.2 carries **only external reference data** — neither
  the B200 nor the Dell MI355X figure was measured on this host.
- **Part A leftovers**: `iet` and `rvs_level_5` hit their timeouts (rc=124) and
  `rvs_level_4` core-dumped. Not retried.
- **Single runs.** Almost everything here is n=1. The repeat experiment tests only the MAD gap.
- **`pebb` contamination.** Part A's PCIe bandwidth was measured during heavy NVMe I/O and
  showed NUMA page-allocation errors; the recheck is queued but its result is not yet in.

---

## 6. Files

| File | Contents |
|---|---|
| `rvs_tflops.{md,csv}` | Part A — TFLOPS × 9 precisions × N=1..8, Dell/AMD/B200 comparison |
| `fp4_investigation.md` | Part A — fp4 N≥5 scaling anomaly (non-deterministic) |
| `rccl.{md,csv}`, `rccl_busbw.png` | Part B — collective bandwidth, cliff analysis |
| `PRIMUS_REPORT.md` | Part C — GEMM/attention/RCCL microbenches + Megatron |
| `atom.{md,csv}` | Part D — three model tiers at a glance |
| `kimi-k3-improve.md` | **Kimi-K3 entry point** — all runs consolidated + next steps |
| `kimi-k3-base.md`, `-mad.md`, `-maxseqs.md`, `-comparison.md` | Per-run Kimi-K3 detail |
| `../notes.md`, `../notes-kimi-k3.md` | Host layout; Kimi-K3 rerun rationale and diagnoses |
| `../plan.md` | The benchmark plan, with a setup log of what actually happened |
