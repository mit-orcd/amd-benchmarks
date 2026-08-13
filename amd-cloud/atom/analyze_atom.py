#!/usr/bin/env python3
"""Aggregate ATOM serving sweeps into results/atom.{md,csv}.

Usage: analyze_atom.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>

Reads the per-concurrency c<N>.json files written by atom.benchmarks.benchmark_serving
(vLLM-compatible schema). Reports throughput vs concurrency, TTFT/TPOT percentiles, and
locates the throughput/latency knee -- the concurrency past which added load buys little
throughput but costs latency, which is the number that actually matters for serving.
"""
import argparse, csv, json, re, sys
from pathlib import Path

# Keys emitted by atom/benchmarks/benchmark_serving.py --save-result
FIELDS = ["max_concurrency", "completed", "duration",
          "request_throughput", "output_throughput", "total_token_throughput",
          "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
          "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
          "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
          "total_input_tokens", "total_output_tokens", "model_id"]


def load(d: Path):
    rows = []
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {j.name}: {e}", file=sys.stderr)
            continue
        row = {k: data.get(k) for k in FIELDS}
        if row.get("max_concurrency") in (None, ""):
            m = re.match(r"c(\d+)", j.stem)
            row["max_concurrency"] = int(m.group(1)) if m else None
        row["run"] = d.name
        rows.append(row)
    return rows


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


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
    rows.sort(key=lambda r: (r["run"], r["max_concurrency"] or 0))

    with (a.out / "atom.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run"] + FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    model = next((r.get("model_id") for r in rows if r.get("model_id")), "unknown")
    L = ["# ATOM serving benchmark — MI355X", "",
         f"Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on "
         f"8 x MI355X (gfx950), ROCm 7.14.",
         f"Model: `{model}`  ·  Source runs: {', '.join(sorted({r['run'] for r in rows}))}", "",
         "Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric "
         "bandwidth: a load generator drives an OpenAI-compatible server at fixed "
         "input/output length while concurrency varies. There is no `dell-cloud/` baseline "
         "for this suite -- compare against ATOM's public dashboard, not against this repo.", "",
         "## Throughput and latency vs concurrency", "",
         "| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | "
         "TPOT med (ms) | TPOT p99 (ms) | completed |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append("| " + " | ".join([
            f(r["max_concurrency"], 0), f(r["request_throughput"], 2),
            f(r["output_throughput"]), f(r["total_token_throughput"]),
            f(r["median_ttft_ms"]), f(r["p99_ttft_ms"]),
            f(r["median_tpot_ms"], 2), f(r["p99_tpot_ms"], 2),
            f(r["completed"], 0)]) + " |")

    # ---- knee detection ---------------------------------------------------------
    L += ["", "## Throughput / latency knee", ""]
    ok = [r for r in rows if r.get("output_throughput") and r.get("max_concurrency")]
    if len(ok) >= 2:
        best = max(ok, key=lambda r: r["output_throughput"])
        knee, prev = None, None
        for r in ok:
            if prev and prev["output_throughput"]:
                gain = (r["output_throughput"] - prev["output_throughput"]) / prev["output_throughput"]
                cratio = r["max_concurrency"] / prev["max_concurrency"]
                # doubling concurrency but buying <20% throughput => past the knee
                if cratio >= 1.9 and gain < 0.20 and knee is None:
                    knee = prev
            prev = r
        L += [f"- **Peak output throughput**: {f(best['output_throughput'])} tok/s at "
              f"concurrency {f(best['max_concurrency'], 0)} "
              f"(TTFT med {f(best['median_ttft_ms'])} ms, TPOT med {f(best['median_tpot_ms'], 2)} ms)."]
        if knee:
            L += [f"- **Knee at concurrency {f(knee['max_concurrency'], 0)}**: beyond this, "
                  f"doubling concurrency buys <20% more throughput while latency keeps rising. "
                  f"For a latency-sensitive deployment this is the operating point; past it you "
                  f"are trading TTFT for very little tok/s."]
        else:
            L += ["- No knee detected in the sampled range — throughput was still scaling at the "
                  "highest concurrency tested. Extend the concurrency list to find the ceiling."]
        lo, hi = ok[0], ok[-1]
        if lo["median_tpot_ms"] and hi["median_tpot_ms"]:
            L += [f"- TPOT grows {hi['median_tpot_ms'] / lo['median_tpot_ms']:.1f}x "
                  f"({f(lo['median_tpot_ms'], 2)} -> {f(hi['median_tpot_ms'], 2)} ms) across the "
                  f"sweep while output throughput grows "
                  f"{hi['output_throughput'] / lo['output_throughput']:.1f}x — the batching "
                  f"trade-off in one line."]
    else:
        L += ["- Not enough points to locate a knee."]

    L += ["", "## Metric definitions", "",
          "| Metric | Meaning |", "|---|---|",
          "| TTFT | Time to first token — prefill latency; what a user perceives as 'lag'. |",
          "| TPOT | Time per output token — steady-state decode speed after the first token. |",
          "| ITL | Inter-token latency — per-token gaps, the jitter behind TPOT. |",
          "| output tok/s | Generated tokens per second across all concurrent requests. |",
          "| total tok/s | Input + output tokens per second (prefill work included). |", "",
          "## Caveats", "",
          "- **The load generator is co-located with the server**, competing for host CPU. This "
          "is the normal ATOM/vLLM benchmarking convention but is not a clean client/server "
          "split; absolute req/s at high concurrency is mildly pessimistic.",
          "- `--ignore-eos` forces every request to generate exactly OSL tokens, so throughput "
          "is not skewed by early stopping. Good for comparability, not representative of real "
          "traffic where output lengths vary.",
          "- `--random-range-ratio` jitters prompt lengths around ISL, so prefix caching cannot "
          "trivially inflate results.",
          "- KV cache dtype is fp8 by default (`KV_CACHE_DTYPE`), which affects both memory "
          "headroom and achievable concurrency. Record it when comparing runs.", "",
          "## Source data", "", "| What | Where |", "|---|---|",
          "| Per-concurrency JSON | `logs/atom/sweep_*/c<N>.json` |",
          "| Per-concurrency log | `logs/atom/sweep_*/c<N>.log` |",
          "| Sweep summary | `logs/atom/sweep_*/summary.txt` |",
          "| Server log | `logs/atom/server_*/atom_server.log` |",
          "| This table as CSV | `results/atom.csv` |", ""]

    (a.out / "atom.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/atom.md and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
