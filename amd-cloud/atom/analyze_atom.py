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
         "| Tier | Model | TP | Peak tok/s | @ conc | TTFT med @c=1 (ms) | TPOT med @c=1 (ms) | Knee |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in order:
        rs = models[m]
        tier, tp = tier_of(m)
        best = max(rs, key=lambda r: r.get("output_throughput") or 0)
        lo = min(rs, key=lambda r: r["max_concurrency"])
        k = knee(rs)
        L.append(f"| {tier} | `{m}` | {tp or '—'} | **{f(best['output_throughput'])}** | "
                 f"{f(best['max_concurrency'],0)} | {f(lo['median_ttft_ms'])} | "
                 f"{f(lo['median_tpot_ms'],2)} | "
                 f"{f(k['max_concurrency'],0) if k else 'none in range'} |")
    L += [""]

    # cross-model interpretation, computed rather than asserted
    if len(order) >= 2:
        peaks = {m: max((r.get("output_throughput") or 0) for r in models[m]) for m in order}
        q = next((m for m in order if m.startswith("Qwen3-8B")), None)
        l70 = next((m for m in order if m.startswith("Llama-3.1-70B")), None)
        kimi = next((m for m in order if m.startswith("Kimi-K3")), None)
        notes = []
        if q and l70 and peaks[q] and peaks[l70]:
            notes.append(
                f"- **8B (TP=1) vs 70B (TP=8): {peaks[q]/peaks[l70]:.2f}x.** An ~8.75x larger "
                f"model costs only ~{peaks[q]/peaks[l70]:.1f}x throughput, because TP=8 "
                f"spreads it across the whole box. Note the 8B runs on **one** GPU and the "
                f"70B on **eight**, so this is a serving-capability comparison, not a "
                f"per-GPU efficiency one.")
        if l70 and kimi and peaks[kimi]:
            notes.append(
                f"- **70B vs Kimi-K3: {peaks[l70]/peaks[kimi]:.1f}x.** Both TP=8. Kimi-K3 is "
                f"2.78T total parameters but only ~84B *active* per token (MoE, top-16 of "
                f"896 experts), so its active size is comparable to the 70B — yet it is "
                f"{peaks[l70]/peaks[kimi]:.1f}x slower. That is the real cost of MoE: sparse "
                f"activation saves FLOPs but not weight *traffic*, and traffic is the "
                f"binding constraint. See `kimi-k3.md` for the full bandwidth analysis.")
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
