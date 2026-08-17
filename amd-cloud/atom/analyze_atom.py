#!/usr/bin/env python3
"""Aggregate ATOM serving sweeps into results/atom.{md,csv}.

Usage: analyze_atom.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>

One section per model, plus a cross-model comparison. Grouping by model matters: the
per-concurrency JSON records `model_id` as the container mount path (`/model`) for every
tier, so without recovering the real name from the sweep's summary.txt all three tiers
collapse into one undifferentiated table and the knee detection compares an 8B model's
latency against a 2.78T model's.
"""
import argparse, csv, json, re, sys
from pathlib import Path

FIELDS = ["max_concurrency", "output_throughput", "total_token_throughput",
          "request_throughput", "median_ttft_ms", "p99_ttft_ms",
          "median_tpot_ms", "p99_tpot_ms", "completed"]

# Tier labels, so the report reads in the order the plan defines rather than by timestamp.
TIER_HINTS = [("Qwen3-8B", "tier 1", 1), ("Llama-3.1-70B", "tier 2", 8), ("Kimi-K3", "tier 3", 8)]

# Per-model facts, read from each checkpoint's config.json and its on-disk size.
# `params` is total; `active` is what actually fires per token (differs only for MoE).
MODEL_INFO = {
    "Qwen3-8B-FP8": dict(
        hf="Qwen/Qwen3-8B-FP8", arch="Qwen3ForCausalLM", kind="dense",
        params="8 B", active="8 B", disk="8.9 GB", quant="FP8 (block 128)", weight_gb=8.0,
        layers=36, hidden=4096, heads=32, kv=8, vocab=151936, ctx="40K",
        note="Smallest tier and the only single-GPU run. Ungated on HF, so it proves the "
             "serving path without a token. GQA 32:8."),
    "Llama-3.1-70B-Instruct-FP8": dict(
        hf="RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8", arch="LlamaForCausalLM", kind="dense",
        params="70 B", active="70 B", disk="68 GB", quant="FP8 W8A8 (compressed-tensors)", weight_gb=68.0,
        layers=80, hidden=8192, heads=64, kv=8, vocab=128256, ctx="131K",
        note="The dense headline. At TP=8 every layer all-reduces, so this is the tier that "
             "puts RCCL in the per-token critical path. RedHatAI quant chosen because "
             "meta-llama is gated."),
    "Kimi-K3": dict(
        hf="moonshotai/Kimi-K3", arch="KimiK3ForConditionalGeneration", kind="MoE (hybrid attn)",
        params="2.78 T", active="~84 B", disk="1.5 TB", quant="MXFP4 experts + PTPC-FP8 rest",
        layers=93, hidden=7168, heads=96, kv=96, vocab=163840, ctx="1M",
        # Weight bytes actually READ per decode step at the peak concurrency measured.
        # For MoE this is NOT the active-parameter size: at batch 64 the 64 tokens route
        # independently, so E*(1-(1-1/E)^(B*topk)) = 610 of 896 experts fire per layer.
        weight_gb=931.0,
        note="Frontier MoE: 896 routed experts, top-16 + 2 shared, so only ~3% of the model "
             "fires per token. 24 MLA full-attention layers + 69 KDA linear-attention "
             "layers — only the 24 keep a growing KV cache."),
}


def model_of(d: Path) -> str:
    """Recover the real model name; the JSON only has the container path."""
    s = d / "summary.txt"
    if s.exists():
        m = re.search(r"models/([A-Za-z0-9._-]+)", s.read_text(errors="replace"))
        if m:
            return m.group(1)
    return d.name


def tier_of(model: str):
    for hint, tier, tp in TIER_HINTS:
        if model.startswith(hint):
            return tier, tp
    return "—", None


def load(d: Path):
    rows = []
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        r = {k: data.get(k) for k in FIELDS}
        if not r.get("max_concurrency"):
            m = re.match(r"c(\d+)", j.stem)
            r["max_concurrency"] = int(m.group(1)) if m else None
        r["run"], r["model"] = d.name, model_of(d)
        rows.append(r)
    return rows


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def knee(rows):
    """Concurrency past which doubling load buys <20% throughput."""
    ok = [r for r in rows if r.get("output_throughput") and r.get("max_concurrency")]
    ok.sort(key=lambda r: r["max_concurrency"])
    prev = None
    for r in ok:
        if prev and prev["output_throughput"]:
            gain = (r["output_throughput"] - prev["output_throughput"]) / prev["output_throughput"]
            if r["max_concurrency"] / prev["max_concurrency"] >= 1.9 and gain < 0.20:
                return prev
        prev = r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in a.dirs:
        rows += load(d)
    if not rows:
        sys.exit("no ATOM benchmark JSON found")

    with (a.out / "atom.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "run"] + FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # group by model, ordered by tier
    models = {}
    for r in rows:
        models.setdefault(r["model"], []).append(r)
    order = sorted(models, key=lambda m: (tier_of(m)[0], m))

    L = ["# ATOM serving benchmark — MI355X", "",
         "Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on "
         "8 x MI355X (gfx950), ROCm 7.14. Workload: ISL/OSL 1024/1024, `--ignore-eos`, "
         "saturating request rate.", "",
         "Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric "
         "bandwidth. There is no `dell-cloud/` baseline for this suite — compare against "
         "ATOM's public dashboard, not against this repo.", "",
         "## Summary — three models at a glance", "",
         "| Tier | Model | Params (total / active) | On disk | TP | Peak tok/s | @ conc | "
         "TTFT med @c=1 (ms) | TPOT med @c=1 (ms) | Knee |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in order:
        rs = models[m]
        tier, tp = tier_of(m)
        info = MODEL_INFO.get(m, {})
        best = max(rs, key=lambda r: r.get("output_throughput") or 0)
        lo = min(rs, key=lambda r: r["max_concurrency"])
        k = knee(rs)
        size = (f"{info['params']} / {info['active']}" if info.get("params") else "—")
        L.append(f"| {tier} | `{m}` | {size} | {info.get('disk','—')} | {tp or '—'} | "
                 f"**{f(best['output_throughput'])}** | "
                 f"{f(best['max_concurrency'],0)} | {f(lo['median_ttft_ms'])} | "
                 f"{f(lo['median_tpot_ms'],2)} | "
                 f"{f(k['max_concurrency'],0) if k else 'none in range'} |")
    L += ["",
          "*Active* params are what actually fire per token — identical to total for a dense "
          "model, but only ~3% of total for Kimi-K3's MoE. That distinction drives most of "
          "the throughput differences below.", ""]

    # ---- model descriptions ----
    L += ["## The three models", ""]
    for m in order:
        info = MODEL_INFO.get(m)
        tier, tp = tier_of(m)
        if not info:
            L += [f"### {tier.title()} — `{m}`", "", "_No architecture record._", ""]
            continue
        L += [f"### {tier.title()} — `{m}`", "",
              f"[`{info['hf']}`](https://huggingface.co/{info['hf']}) · `{info['arch']}` · "
              f"**{info['kind']}**", "",
              "| | |", "|---|---|",
              f"| Parameters | **{info['params']}** total, {info['active']} active per token |",
              f"| Checkpoint on disk | {info['disk']} |",
              f"| Quantization | {info['quant']} |",
              f"| Layers / hidden | {info['layers']} / {info['hidden']} |",
              f"| Attn heads / KV heads | {info['heads']} / {info['kv']} |",
              f"| Vocab / max context | {info['vocab']:,} / {info['ctx']} |",
              f"| Tensor parallel | TP={tp} |", "",
              info["note"], ""]

    # cross-model interpretation, computed rather than asserted
    if len(order) >= 2:
        peaks = {m: max((r.get("output_throughput") or 0) for r in models[m]) for m in order}
        q = next((m for m in order if m.startswith("Qwen3-8B")), None)
        l70 = next((m for m in order if m.startswith("Llama-3.1-70B")), None)
        kimi = next((m for m in order if m.startswith("Kimi-K3")), None)
        notes = []
        if q and l70 and peaks[q] and peaks[l70]:
            pg_q, pg_l = peaks[q] / 1, peaks[l70] / 8
            notes.append(
                f"- **8B vs 70B — tracks model size, roughly.** Raw throughput differs only "
                f"{peaks[q]/peaks[l70]:.2f}x, but that hides the GPU count: **per GPU** it is "
                f"{pg_q:,.0f} vs {pg_l:,.0f} tok/s, a **{pg_q/pg_l:.1f}x** gap against an "
                f"8.8x active-parameter ratio. So the 70B recovers most of what its size "
                f"costs by using 8 GPUs; the residual ~1.5x beyond pure size scaling is TP "
                f"communication, a larger KV cache per token, and lower per-GPU efficiency. "
                f"That is the expected shape.")
        if l70 and kimi and peaks[kimi]:
            pg_l, pg_k = peaks[l70] / 8, peaks[kimi] / 8
            notes.append(
                f"- **70B vs Kimi-K3 — does NOT track model size, and that is the finding.** "
                f"Both are TP=8, so per GPU it is {pg_l:,.0f} vs {pg_k:,.0f} tok/s — "
                f"**{pg_l/pg_k:.1f}x** apart for only a **1.2x** difference in *active* "
                f"parameters (70 B vs ~84 B). Size does not explain it; **weight traffic** "
                f"does. At batch 64 Kimi's tokens route independently across 896 experts, so "
                f"610 of them fire per layer and the engine reads **931 GB per step vs the "
                f"70B's 68 GB — 13.7x more traffic for 1.2x more active parameters.** A "
                f"{pg_l/pg_k:.1f}x slowdown from 13.7x more traffic is actually *better* than "
                f"linear, because Kimi converts bandwidth to tokens more efficiently (29% of "
                f"roofline vs 4%). This is the central cost of MoE: sparse activation saves "
                f"FLOPs but not bytes, and bytes are the binding constraint. Full analysis "
                f"in `kimi-k3.md`.")
        # ---- roofline: are these numbers what the hardware should give? ----
        HBM = 8000.0   # GB/s per MI355X
        roof_rows = []
        for m in order:
            info = MODEL_INFO.get(m) or {}
            wgb = info.get("weight_gb")
            if not wgb:
                continue
            tier, tp = tier_of(m)
            best = max(models[m], key=lambda r: r.get("output_throughput") or 0)
            B = best["max_concurrency"]
            tps = best["output_throughput"]
            per_gpu = tps / (tp or 1)
            step_ms = B / tps * 1000
            roof = B * (HBM * (tp or 1)) / wgb    # weight-read-bound ceiling
            roof_rows.append((m, tp, B, tps, per_gpu, step_ms, wgb, roof, 100 * tps / roof))
        if roof_rows:
            L += ["### Are these numbers what the hardware should give?", "",
                  "**Roofline tok/s** = the most tokens/s possible if reading weights from "
                  "HBM were the *only* cost. One decode step emits `batch` tokens and must "
                  "read every weight it activates once, so:", "",
                  "```",
                  "step_time >= weight_bytes / (HBM_BW_per_GPU x GPUs)",
                  "roofline  =  batch / step_time",
                  "          =  batch x (HBM_BW_per_GPU x GPUs) / weight_bytes",
                  "```", "",
                  "e.g. Llama-70B: 256 x (8000 GB/s x 8) / 68 GB = **240,941 tok/s**.", "",
                  "`weight_bytes` is what is *actually read*, not model size — identical for "
                  "dense models, but for Kimi-K3 it is **931 GB**, not 1.5 TB and not the "
                  "~84 B active params, because at batch 64 roughly 610 of 896 experts fire "
                  "per layer. It deliberately ignores KV reads, compute, collectives and "
                  "prefill, so it is a *loose upper bound*: far below it means weight traffic "
                  "is not the constraint; near it means it is.", "",
                  "| Model | GPUs | Peak tok/s | tok/s **per GPU** | step (ms) | weights read/step | roofline tok/s | % of roofline |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for (m, tp, B, tps, per_gpu, step_ms, wgb, roof, pct) in roof_rows:
                L.append(f"| `{m}` | {tp} | {f(tps)} | **{f(per_gpu)}** | {step_ms:.1f} | "
                         f"{f(wgb)} GB | {f(roof)} | **{pct:.1f}%** |")
            L += ["",
                  "**Yes — and the % column is the interesting part.** Kimi-K3 sits at ~29% "
                  "of its weight-bandwidth ceiling while the two dense models sit at 4-6%. "
                  "That is not Kimi doing better; it means Kimi is genuinely "
                  "**bandwidth-bound** while Qwen and Llama are not. It also matches, "
                  "independently, the ~29% HBM utilization measured in `kimi-k3.md` §3 — two "
                  "different routes to the same number.", "",
                  "To be precise about *which* bandwidth: this is **intra-GPU HBM** — each "
                  "GPU reading weights out of its own 8 TB/s on-package memory. It is **not** "
                  "the XGMI GPU-to-GPU interconnect, which in the same run carries only "
                  "activation all-reduces and sits at ~1% utilized. The two are often "
                  "conflated; here they differ by roughly 390:1 in traffic. "
                  "**See [`kimi-k3.md`](kimi-k3.md) for the full breakdown** — §3 ranks the "
                  "three candidate bottlenecks (compute 1.1%, HBM ~29%, XGMI ~1.1%), §4 "
                  "gives the per-step byte volumes on each path, and the terminology section "
                  "at the top defines HBM vs XGMI.", "",
                  "#### If the dense models are not bandwidth-bound, what limits them?", "",
                  "**Not compute either.** Accounting for a Llama-70B decode step at c=256 "
                  "(measured step time 27.4 ms):", "",
                  "| Candidate cost | Estimated per step | Share of step |",
                  "|---|---:|---:|",
                  "| Weight reads (68 GB / 8 GPUs at 8 TB/s) | ~1.1 ms | ~4% |",
                  "| KV-cache reads (~7.9 GB/GPU) | ~1.0 ms | ~4% |",
                  "| Decode compute (2 x 70e9 x 256 FLOP) | ~3.4 ms | ~12% |",
                  "| TP all-reduce (160 calls/step) | ~2-3 ms | ~10% |",
                  "| **Unaccounted** | **~19 ms** | **~70%** |", "",
                  "No single hardware resource is saturated: bandwidth ~4%, compute ~12%, "
                  "interconnect ~10%. Calling these models \"compute-bound\" would be wrong. "
                  "The ~70% residual is almost certainly **prefill interleaved with decode**, "
                  "plus per-step scheduler overhead.", "",
                  "**What prefill and decode are.** Serving a request has two phases. "
                  "*Prefill* is the one-time pass over the whole input prompt — all 1024 "
                  "tokens processed in a single parallel forward pass, compute-heavy, once "
                  "per request. *Decode* is everything after: output tokens generated one at "
                  "a time, each a separate forward pass reading the KV cache. All the "
                  "throughput and TPOT numbers here measure decode, and the roofline above "
                  "models decode only.", "",
                  "**Why prefill becomes the limiter.** ATOM (like vLLM) uses continuous "
                  "batching with chunked prefill: prefill and decode share the same GPU time "
                  "slice, and a request cannot decode until it is prefilled. At c=256 with "
                  "ISL=1024 that is **~262,000 prompt tokens** of backlog competing for the "
                  "same GPU. While the scheduler runs prefill chunks, decode steps for "
                  "in-flight requests wait. The signature is visible in the data: TTFT "
                  "**median stays low (182 ms) while p99 blows out to 5,470 ms** — most "
                  "requests prefill quickly, the tail queues behind the backlog.", "",
                  "**\"Scheduling\"** is the per-step cost of the serving loop itself — "
                  "choosing which requests enter this step's batch, KV-cache block "
                  "allocation, continuous-batching bookkeeping. It is CPU-side work on the "
                  "critical path of every step, independent of GPU load.", "",
                  "**Why Kimi-K3 escapes this.** Its concurrency is capped at 64, so the "
                  "prefill backlog is far smaller — and its per-step GPU work (931 GB of "
                  "weight reads) is so large that prefill and scheduler overhead are "
                  "comparatively negligible. That is much of why its roofline utilization "
                  "(29%) looks so much healthier than the dense models' (4-6%): for Kimi the "
                  "GPU-bound part genuinely dominates the step, while for Qwen and Llama the "
                  "GPU-bound part is small and everything else fills the remaining ~70%.", "",
                  "So the three tiers are limited by three different things: **Kimi-K3 by "
                  "memory bandwidth**, and **Qwen / Llama by prefill throughput and "
                  "scheduling**, with no hardware unit near its ceiling. Their distance from "
                  "the roofline is therefore expected rather than a defect — a *pure-decode* "
                  "roofline is the wrong yardstick for a **mixed prefill+decode** serving "
                  "benchmark.", "",
                  "> **This last part is inference, not measurement.** The residual is what "
                  "is left after subtracting four estimated costs; it is not a profiled "
                  "breakdown, and the individual estimates carry their own error. The TTFT "
                  "median-vs-p99 gap is real measured evidence that queueing happens, but "
                  "the *split* between prefill contention and scheduler overhead inside that "
                  "~70% is not measured — no trace shows \"X ms prefill, Y ms scheduler\". "
                  "Settling it needs a profiler trace, or a decode-only run (short ISL, or "
                  "prefill and decode measured separately).", ""]
        if notes:
            L += ["### Reading the comparison", ""] + notes + [""]
        L += ["> **Caveat: different TP.** Tier 1 is TP=1 (single GPU), tiers 2 and 3 are "
              "TP=8. Throughput is therefore not normalized per GPU, and the tiers answer "
              "\"what can this box serve for this model\" rather than \"which model is more "
              "efficient per GPU\".", ""]

    # per-model detail
    for m in order:
        rs = sorted(models[m], key=lambda r: r["max_concurrency"])
        tier, tp = tier_of(m)
        L += [f"## {tier.title()} — `{m}` (TP={tp or '—'})", "",
              "| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | "
              "TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rs:
            L.append("| " + " | ".join([
                f(r["max_concurrency"], 0), f(r["request_throughput"], 2),
                f(r["output_throughput"]), f(r["total_token_throughput"]),
                f(r["median_ttft_ms"]), f(r["p99_ttft_ms"]),
                f(r["median_tpot_ms"], 2), f(r["p99_tpot_ms"], 2),
                f(r["completed"], 0)]) + " |")
        k = knee(rs)
        lo, hi = rs[0], rs[-1]
        best = max(rs, key=lambda r: r.get("output_throughput") or 0)
        L += ["",
              f"- Peak **{f(best['output_throughput'])} tok/s** at concurrency "
              f"{f(best['max_concurrency'],0)}."]
        if k:
            L.append(f"- **Knee at concurrency {f(k['max_concurrency'],0)}** — beyond this, "
                     f"doubling load buys <20% more throughput while latency keeps climbing. "
                     f"For latency-sensitive serving this is the operating point.")
        else:
            L.append("- **No knee in the sampled range** — still scaling at the highest "
                     "concurrency tested; the ceiling is set by `max_num_seqs`, not by "
                     "saturation.")
        if lo["median_tpot_ms"] and hi["median_tpot_ms"] and lo["output_throughput"]:
            L.append(f"- Across the sweep TPOT grows "
                     f"{hi['median_tpot_ms']/lo['median_tpot_ms']:.1f}x "
                     f"({f(lo['median_tpot_ms'],2)} -> {f(hi['median_tpot_ms'],2)} ms) while "
                     f"throughput grows "
                     f"{hi['output_throughput']/lo['output_throughput']:.1f}x — the batching "
                     f"trade-off for this model.")
        L += [""]

    L += ["## Metric definitions", "", "| Metric | Meaning |", "|---|---|",
          "| TTFT | Time to first token — prefill latency; the user-perceived lag. |",
          "| TPOT | Time per output token — steady-state decode speed after the first token. |",
          "| output tok/s | Generated tokens/s across all concurrent requests. |",
          "| total tok/s | Input + output tokens/s (prefill work included). |", "",
          "## Caveats", "",
          "- **The load generator is co-located with the server**, competing for host CPU. "
          "Standard ATOM/vLLM practice, but not a clean client/server split; req/s at high "
          "concurrency is mildly pessimistic.",
          "- `--ignore-eos` forces exactly OSL tokens per request, so throughput is not "
          "skewed by early stopping — comparable, but not representative of real traffic.",
          "- `--random-range-ratio 0.8` jitters prompt lengths so prefix caching cannot "
          "inflate results.",
          "- KV cache dtype is fp8; Kimi-K3 additionally runs with prefix caching disabled "
          "(required — KDA recurrent state cannot be rebuilt from the paged cache).", "",
          "## Deep dive", "",
          "`kimi-k3.md` analyses tier 3 in detail: achieved TFLOP/s, the GPU memory "
          "breakdown (weights vs KV pool), why the workload is HBM-bandwidth-bound rather "
          "than compute- or interconnect-bound, and the intra-GPU vs intra-node data "
          "volumes.", "",
          "## Source data", "", "| What | Where |", "|---|---|",
          "| Per-concurrency JSON / logs | `logs/atom/sweep_*/c<N>.{json,log}` |",
          "| Sweep summaries | `logs/atom/sweep_*/summary.txt` |",
          "| Server logs | `logs/atom/server_*/atom_server.log` |",
          "| This table as CSV | `results/atom.csv` |", ""]

    (a.out / "atom.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/atom.md and .csv ({len(rows)} rows, {len(order)} models)")


if __name__ == "__main__":
    main()
