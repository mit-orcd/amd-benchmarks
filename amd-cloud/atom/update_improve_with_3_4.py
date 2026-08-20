#!/usr/bin/env python3
"""Fold next-step #3 (ISL=4096) and #4 (repeats) results into results/kimi-k3-improve.md.

Usage: update_improve_with_3_4.py --isl <dir|none> --repeats <dir|none> <improve.md>

Idempotent via HTML comment markers. Each section is written only if its data exists, so a
partial queue (one stage failed) still records what succeeded instead of writing nothing.
Also strikes through the corresponding entries in "Ranked next experiments" so the list
reflects reality rather than leaving stale instructions.
"""
import argparse, json, re, statistics as st, sys
from pathlib import Path

BEGIN = "<!-- BEGIN next-steps-3-4 (auto-generated) -->"
END = "<!-- END next-steps-3-4 -->"

# Run C reference at ISL=1024 (same cap 256), for the ISL comparison.
RUN_C_1024 = {64: 1237.0, 128: 1792.5, 256: 2482.2}


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def load_sweep(d: Path):
    rows = []
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("output_throughput"):
            rows.append(dict(c=data.get("max_concurrency") or int(re.sub(r"\D", "", j.stem) or 0),
                             tps=data["output_throughput"],
                             ttft=data.get("median_ttft_ms"),
                             tpot=data.get("median_tpot_ms")))
    rows.sort(key=lambda r: r["c"])
    return rows


def load_repeats(d: Path):
    out = {}
    for j in sorted(d.glob("*_rep*.json")):
        m = re.match(r"(.+)_rep(\d+)\.json$", j.name)
        if not m:
            continue
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("output_throughput"):
            out.setdefault(m.group(1), []).append(data["output_throughput"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=str, default="none")
    ap.add_argument("--repeats", type=str, default="none")
    ap.add_argument("md", type=Path)
    a = ap.parse_args()

    L = [BEGIN, ""]
    wrote = []

    # ---- #3 ISL=4096 ----
    isl_rows = []
    if a.isl != "none" and Path(a.isl).is_dir():
        isl_rows = load_sweep(Path(a.isl))
    if isl_rows:
        wrote.append("#3")
        L += ["### ISL = 4096 (next-step #3, completed)", "",
              f"Same config as Run C (cap 256) with **input length 4096 instead of 1024** — "
              f"ISL is the only variable. Source: `logs/atom/{Path(a.isl).name}/`, detail in "
              f"`kimi-k3-isl4096.md`.", "",
              "| Concurrency | ISL 1024 (Run C) | **ISL 4096** | 4096/1024 | TTFT med (ms) | TPOT med (ms) |",
              "|---:|---:|---:|---:|---:|---:|"]
        for r in isl_rows:
            ref = RUN_C_1024.get(r["c"])
            ratio = f"**{r['tps']/ref:.2f}×**" if ref else "—"
            L.append(f"| {r['c']} | {f(ref) if ref else '—'} | **{f(r['tps'])}** | {ratio} | "
                     f"{f(r['ttft'])} | {f(r['tpot'],2)} |")
        L += [""]
        best = max(isl_rows, key=lambda r: r["tps"])
        ref = RUN_C_1024.get(best["c"])
        if ref:
            ch = best["tps"] / ref
            if ch < 0.85:
                L += [f"**4× longer prompts cost {(1-ch)*100:.0f}% throughput** at c={best['c']}. "
                      f"Prefill work scales with ISL and competes with decode for the same GPU, "
                      f"so a bigger share of each step goes to prompt processing. Expected "
                      f"direction; the magnitude is the useful number for capacity planning.", ""]
            elif ch > 1.05:
                L += [f"**Throughput *rose* {(ch-1)*100:.0f}% at ISL=4096** (c={best['c']}), which "
                      f"is not the naive expectation. Longer prompts mean more tokens processed "
                      f"per prefill pass, and prefill is compute-dense and efficient — so total "
                      f"token throughput can improve even as per-request latency grows. Check "
                      f"the TTFT column before reading this as a free win.", ""]
            else:
                L += [f"**Throughput is essentially unchanged** ({ch:.2f}× at c={best['c']}) "
                      f"despite 4× the prompt length. That suggests decode dominates the step "
                      f"at this concurrency and prefill is not yet the constraint.", ""]

    # ---- #4 repeats ----
    reps = {}
    if a.repeats != "none" and Path(a.repeats).is_dir():
        reps = load_repeats(Path(a.repeats))
    if reps:
        wrote.append("#4")
        L += ["### Repeatability of the MAD gap (next-step #4, completed)", "",
              f"Three repeats at c=64 per config, to test whether the single-run **0.91×** "
              f"figure in `kimi-k3-comparison.md` is real. Source: "
              f"`logs/atom/{Path(a.repeats).name}/`, detail in `kimi-k3-repeats.md`.", "",
              "| Config | n | mean tok/s | stdev | rel. spread |", "|---|---:|---:|---:|---:|"]
        stats = {}
        for tag in sorted(reps):
            v = reps[tag]
            mean = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
            spread = (max(v) - min(v)) / mean * 100 if mean else 0
            stats[tag] = (mean, sd, max(v) - min(v))
            L.append(f"| `{tag}` | {len(v)} | **{f(mean)}** | {f(sd,1)} | {spread:.1f}% |")
        L += [""]
        at = next((t for t in stats if t.startswith("A")), None)
        bt = next((t for t in stats if t.startswith("B")), None)
        if at and bt:
            ma, sa, spa = stats[at]; mb, sb, spb = stats[bt]
            gap = abs(ma - mb); noise = max(spa, spb); ratio = mb / ma
            if gap <= noise:
                L += [f"**The ~9% gap does not survive repeats.** Configs differ by "
                      f"{f(gap,1)} tok/s, within the worst within-config spread "
                      f"({f(noise,1)}). Measured ratio {ratio:.3f}×. **Treat the earlier "
                      f"0.91× claim as unproven** — on this evidence the two recipes perform "
                      f"comparably at c=64.", ""]
            elif gap > 2 * noise:
                L += [f"**The gap is real.** {f(gap,1)} tok/s separation exceeds twice the "
                      f"worst within-config spread ({f(noise,1)}). Ratio {ratio:.3f}× vs the "
                      f"0.908× single-run estimate — the original finding stands.", ""]
            else:
                L += [f"**Borderline.** {f(gap,1)} tok/s gap vs {f(noise,1)} worst spread; "
                      f"ratio {ratio:.3f}×. Directionally consistent with the single-run "
                      f"result but not conclusively separated at n=3.", ""]
        L += ["> Repeats share one server process per config, so this bounds "
              "benchmark-to-benchmark variance, not full cold-start variance.", ""]

    if not wrote:
        print("no #3 or #4 data found — nothing to insert")
        return
    L += [END, ""]
    block = "\n".join(L)

    text = a.md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", block,
                      text, flags=re.DOTALL)
    else:
        anchor = re.search(r"\n## 4\. Next steps", text)
        text = (text[:anchor.start()] + "\n" + block + text[anchor.start():]) if anchor \
               else text.rstrip() + "\n\n" + block

    if isl_rows:
        text = text.replace("3. **ISL = 4096.**", "3. ~~**ISL = 4096.**~~ **DONE — see above.**")
    if reps:
        text = text.replace("4. **Repeats for the 9% MAD gap.**",
                            "4. ~~**Repeats for the 9% MAD gap.**~~ **DONE — see above.**")
    a.md.write_text(text)
    print(f"updated {a.md} with {', '.join(wrote)}")


if __name__ == "__main__":
    main()
