# Kimi-K3 — original recipe vs AMD MAD recipe

Two runs of the same model on the same 8 × MI355X box, differing only in image and launch
configuration. Neither file is modified by this one; it reads both and compares.

| | Run A — original | Run B — AMD MAD recipe |
|---|---|---|
| Image | `rocm/atom-dev:latest` | `rocm/atom-dev:rocm7.2.4_..._20260727_kimi_k3` |
| Recipe source | ATOM in-repo `recipes/Kimi-K3.md` | [ROCm/MAD](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md) |
| Kernel-selection env vars | none | 11 MAD vars (`ATOM_USE_TRITON_GEMM=1`, `ATOM_USE_TRITON_MOE=0`, `AITER_FLYDSL_FORCE=1`, `ATOM_USE_UNIFIED_ATTN=1`, `ATOM_FORCE_ATTN_TRITON=1`, …) |
| `--online_quant_config` | PTPC-FP8 explicit | omitted (MAD relies on env-var kernel selection) |
| `--max-num-batched-tokens` | 16384 | 10240 |
| `--max-num-seqs` | 64 | 64 |
| Concurrency swept | 1 → 64 | 64 → 256 |
| Detail | `kimi-k3-base.md` | `kimi-k3-mad.md` |
| Raw data | `logs/atom/sweep_20260814_164903/` | `logs/atom/kimi_mad_20260818_223148/` |

---

## 1. Head-to-head at matched concurrency (c=64)

The only directly comparable point — both runs measured it.

| Metric | Run A (original) | Run B (MAD) | B / A |
|---|---:|---:|---:|
| Output throughput (tok/s) | **1,258.5** | 1,142.7 | **0.91×** |
| TTFT median (ms) | 285.6 | 281.1 | 0.98× |
| TPOT median (ms) | 49.91 | 52.97 | 1.06× |
| Requests completed | 640 | 640 | — |

**AMD's own validated recipe came out ~9% slower**, with correspondingly higher TPOT. That is
the opposite of the expectation going in — the MAD-pinned image and kernel-selection env vars
were adopted precisely because they should represent AMD's best-tuned path.

### How much confidence to put on that 9%

Moderate, not high. Honest caveats:

- **Single run each, no repeats.** No error bars. A 9% gap could be partly run-to-run
  variance; nothing here establishes it is reproducible.
- **Two variables moved together**, not one: the image changed (`latest` →
  `rocm7.2.4_...kimi_k3`) *and* the launch config changed (env vars, quant config,
  batched-tokens). This comparison cannot attribute the difference to either alone.
- **Run B required a patch to work at all** — see §3. The `flash-linear-attention` package
  was pip-installed into the container at start. If the intended build ships a different or
  tuned `fla`, the installed 0.5.2 may not be what AMD benchmarked against.

What can be said safely: **the MAD recipe did not improve throughput on this host**, and there
is no evidence in this data for adopting it over the original configuration.

---

## 2. The throughput plateau — the more important finding

Run B swept past `--max-num-seqs 64`, which Run A never did. The result is unambiguous:

| Concurrency | Throughput (tok/s) | TTFT median (ms) | TPOT median (ms) |
|---:|---:|---:|---:|
| 64 | 1,142.7 | **281** | 52.97 |
| 128 | 1,187.3 | **49,456** | 53.35 |
| 256 | 1,182.4 | **149,723** | 53.85 |

Going from concurrency 64 → 256 bought **+3.5% throughput** and cost **533× TTFT**.
Throughput is flat at ~1,180 tok/s; TPOT is flat at ~53 ms. Only the queue grows.

**This is the signature of a hard in-flight cap.** The server admits at most
`max-num-seqs = 64` sequences to the batch regardless of how many clients are waiting, so
every request beyond 64 sits in a queue. Median TTFT of 150 seconds at c=256 is almost
entirely wait time, not model time — it is measuring the queue, not the engine.

Two consequences:

1. **MAD's published sweep (64/128/256) with `max-num-seqs=64` does not characterize serving
   capability past 64.** The c=128 and c=256 rows describe queueing behaviour. They are
   legitimate numbers, but they answer "what happens when you overload a 64-slot server",
   not "how fast is this model at batch 256".
2. **It confirms the prediction in `kimi-k3-base.md` §3.2** that `max-num-seqs` — not hardware —
   is the binding limit. HBM sat at ~29% (Run A) / ~18–26% (Run B); compute ~1%; XGMI ~1%.
   Nothing is saturated except the scheduler's own admission cap.

### Why HBM utilization *fell* in Run B

`kimi-k3-mad.md` reports ~18% versus Run A's ~29%. That is an artifact of where the
number is taken, not a regression in memory throughput: the summary computes it at the
peak-throughput concurrency, which is c=128 in Run B. At c=128 more experts activate
(805 of 896 vs 610), so weight traffic per step rises to 153 GB, but step time rises faster
because the extra tokens are queued rather than batched. Recomputed at matched c=64, Run B is
~26% against Run A's ~29% — consistent with the 9% throughput gap, not a separate effect.

---

## 3. Run B required fixing AMD's image before it would run

The MAD-pinned image is missing `flash-linear-attention` (`fla`), which ATOM's Kimi-K3 model
file imports **unconditionally** for the KDA prefill path (`kimi_k3.py:749`, no flag guard, no
fallback). 69 of Kimi-K3's 93 layers are KDA linear-attention, so without it the server loads,
answers `/v1/models`, and then dies on the first real request:

```
ModuleNotFoundError: No module named 'fla'
```

Fixed by installing it inside the container at startup (guarded and idempotent). Full
diagnosis in `notes-kimi-k3.md`. Worth noting `rocm/atom-dev:latest` — Run A's image — does
ship it, so this is a regression in the dated MAD tag rather than a general ATOM issue.

---

## 4. What this does not test

- **`max-num-seqs` raised.** Both runs cap in-flight sequences at 64. The plateau in §2 says
  that is the binding constraint, so the obvious next experiment is raising it and re-sweeping
  — that is the one variable most likely to move throughput, and neither run varies it.
- **ISL = 4096.** MAD's ATOM table specifies input lengths 1024 **and** 4096; both runs here
  use 1024 only. A longer prompt shifts the prefill/decode balance and would likely change the
  picture at high concurrency.
- **Repeats.** One run per configuration. Nothing here supports claims about small differences
  being reproducible.
- **Isolating image from config.** Would need Run A's config on Run B's image, or vice versa.

---

## Bottom line

| Question | Answer from this data |
|---|---|
| Does AMD's MAD recipe beat the original config? | **No** — ~9% slower at matched concurrency (single run, caveats in §1) |
| Does sweeping concurrency past 64 help? | **No** — +3.5% throughput, 533× TTFT; it measures queueing |
| What actually limits this workload? | `max-num-seqs=64`, not HBM (~29%), compute (~1%) or XGMI (~1%) |
| Is AMD's published image usable as-is? | **No** — missing `fla`, cannot serve Kimi-K3 without patching |
