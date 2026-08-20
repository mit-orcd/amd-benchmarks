#!/usr/bin/env python3
"""Generate results/kimi-k3-repeats.md — does the ~9% MAD-vs-original gap survive repeats?

Usage: analyze_repeats.py <kimi_repeats_dir> -o <results_dir>

The single-run comparison in kimi-k3-comparison.md reported MAD at 0.91x the original and
explicitly flagged that one run per config cannot support a claim that size. This computes
mean/stdev/spread per config and states whether the gap is larger than the observed noise.
"""
import argparse, csv, json, re, statistics as st, sys
from pathlib import Path


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def load(d: Path):
    """{config_tag: [rows]} from <tag>_rep<N>.json"""
    out = {}
    for j in sorted(d.glob("*_rep*.json")):
        m = re.match(r"(.+)_rep(\d+)\.json$", j.name)
        if not m:
            continue
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("output_throughput"):
            continue
        out.setdefault(m.group(1), []).append(dict(
            rep=int(m.group(2)), tps=data["output_throughput"],
            ttft=data.get("median_ttft_ms"), tpot=data.get("median_tpot_ms"),
            completed=data.get("completed")))
    for k in out:
        out[k].sort(key=lambda r: r["rep"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = load(a.sweep)
    if not cfg:
        sys.exit(f"no repeat results in {a.sweep}")

    with (a.out / "kimi-k3-repeats.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["config", "rep", "tok_s", "ttft_ms", "tpot_ms", "completed"])
        for tag, rows in cfg.items():
            for r in rows:
                w.writerow([tag, r["rep"], r["tps"], r["ttft"], r["tpot"], r["completed"]])

    L = ["# Kimi-K3 — repeatability of the MAD-vs-original gap", "",
         "`kimi-k3-comparison.md` reported the MAD recipe at **0.91× the original** from a",
         "single run each, and flagged that one run per config cannot support a claim that",
         "size. This run repeats c=64 on both configs to separate signal from noise.", "",
         f"Source: `logs/atom/{a.sweep.name}/`", "",
         "## Per-repeat measurements", "",
         "| Config | Rep | tok/s | TTFT med (ms) | TPOT med (ms) | completed |",
         "|---|---:|---:|---:|---:|---:|"]
    for tag in sorted(cfg):
        for r in cfg[tag]:
            L.append(f"| `{tag}` | {r['rep']} | {f(r['tps'])} | {f(r['ttft'])} | "
                     f"{f(r['tpot'],2)} | {f(r['completed'],0)} |")
    L += ["", "## Statistics", "",
          "| Config | n | mean tok/s | stdev | spread (max−min) | rel. spread |",
          "|---|---:|---:|---:|---:|---:|"]
    stats = {}
    for tag in sorted(cfg):
        v = [r["tps"] for r in cfg[tag]]
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        spread = max(v) - min(v)
        stats[tag] = (mean, sd, spread, len(v))
        L.append(f"| `{tag}` | {len(v)} | **{f(mean)}** | {f(sd,1)} | {f(spread,1)} | "
                 f"{100*spread/mean:.1f}% |")
    L += [""]

    a_tag = next((t for t in stats if t.startswith("A")), None)
    b_tag = next((t for t in stats if t.startswith("B")), None)
    if a_tag and b_tag:
        (ma, sa, spa, na), (mb, sb, spb, nb) = stats[a_tag], stats[b_tag]
        ratio = mb / ma
        gap = abs(ma - mb)
        noise = max(spa, spb)
        L += ["## Verdict", "",
              f"- Original (`{a_tag}`): **{f(ma)} ± {f(sa,1)}** tok/s (n={na})",
              f"- MAD (`{b_tag}`): **{f(mb)} ± {f(sb,1)}** tok/s (n={nb})",
              f"- **Ratio MAD/original = {ratio:.3f}×** (single-run estimate was 0.908×)", ""]
        if gap > 2 * noise and noise > 0:
            L += [f"**The gap is real.** The {f(gap,1)} tok/s difference between configs is "
                  f"larger than twice the worst within-config spread ({f(noise,1)} tok/s), so "
                  f"it is not explained by run-to-run variation at this sample size. The "
                  f"single-run 0.91× estimate holds up.", ""]
        elif gap <= noise:
            L += [f"**The gap is NOT distinguishable from noise.** The {f(gap,1)} tok/s "
                  f"difference between configs is within the worst within-config spread "
                  f"({f(noise,1)} tok/s). **The earlier ~9% claim should not be relied on** — "
                  f"on this evidence the two configs perform comparably, and any statement "
                  f"that one is faster needs more repeats or a tighter measurement.", ""]
        else:
            L += [f"**Borderline.** The {f(gap,1)} tok/s gap exceeds the worst within-config "
                  f"spread ({f(noise,1)} tok/s) but not by the 2× margin used here as a "
                  f"threshold. Directionally the original looks faster, but n={na}/{nb} is "
                  f"too small to call it settled. More repeats would resolve it.", ""]

    L += ["## Caveats", "",
          "- **Repeats share a server process.** The model is loaded once per config and the "
          "benchmark run N times against the same live server. This measures "
          "benchmark-to-benchmark variance, **not** full cold-start variance — real "
          "deployment variance (load placement, memory layout, JIT state) could be larger.",
          "- **Single concurrency (c=64).** The gap could differ at other batch sizes; this "
          "tests only the point the original comparison used.",
          "- **Small n.** Three repeats bounds gross noise, not subtle systematic effects.", "",
          "## Source data", "",
          f"| Per-repeat JSON / logs | `logs/atom/{a.sweep.name}/<config>_rep<N>.{{json,log}}` |",
          "|---|---|",
          f"| Server logs | `logs/atom/{a.sweep.name}/<config>_server.log` |",
          "| This table as CSV | `results/kimi-k3-repeats.csv` |",
          "| Single-run comparison this tests | `kimi-k3-comparison.md` |", ""]

    (a.out / "kimi-k3-repeats.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/kimi-k3-repeats.md ({sum(len(v) for v in cfg.values())} runs, "
          f"{len(cfg)} configs)")


if __name__ == "__main__":
    main()
