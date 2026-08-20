#!/usr/bin/env python3
"""Fold the max-num-seqs=512 result (Run D) into results/kimi-k3-improve.md.

Usage: update_improve_with_512.py <run_d_sweep_dir> <kimi-k3-improve.md>

Idempotent via HTML comment markers: re-running replaces the block rather than appending
a second copy. Inserts before "## 4. Next steps" so the result reads as part of the
findings, and marks next-step #1 as completed rather than leaving a stale "to do".

Interprets the outcome against the prediction it was designed to test: Run C's throughput
was still climbing at c=256 with no knee, so either it keeps climbing (cap still binding)
or it flattens (a different limit has taken over). Both are informative; the script says
which happened rather than assuming.
"""
import argparse, json, re, sys
from pathlib import Path

BEGIN = "<!-- BEGIN run-d-maxseqs512 (auto-generated) -->"
END = "<!-- END run-d-maxseqs512 -->"

# Run C reference (max-num-seqs 256), for the like-for-like comparison.
RUN_C = {64: 1237.0, 128: 1792.5, 256: 2482.2}
RUN_C_TTFT = {64: 282.9, 128: 346.4, 256: 463.1}
RUN_C_TPOT = {64: 49.98, 128: 70.98, 256: 103.72}

E, TOPK, SHARED = 896, 16, 2
PER_EXPERT = 3 * 3584 * 3072
MOE_LAYERS, TP, HBM = 92, 8, 8000.0


def weight_gb(B):
    d = E * (1 - (1 - 1 / E) ** (B * TOPK))
    return (d + SHARED) * PER_EXPERT * MOE_LAYERS * 0.5 / 1e9 / TP, d


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("md", type=Path)
    a = ap.parse_args()

    rows = []
    for j in sorted(a.sweep.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            d = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        c = d.get("max_concurrency") or int(re.sub(r"\D", "", j.stem) or 0)
        if d.get("output_throughput"):
            rows.append(dict(c=int(c), tps=d["output_throughput"],
                             ttft=d.get("median_ttft_ms"), tpot=d.get("median_tpot_ms"),
                             done=d.get("completed")))
    if not rows:
        sys.exit(f"no usable results in {a.sweep}")
    rows.sort(key=lambda r: r["c"])
    best = max(rows, key=lambda r: r["tps"])

    L = [BEGIN, "",
         "### Run D — `max-num-seqs 512` (next-step #1, completed)", "",
         f"Ran the top-ranked follow-up: raise the cap again and sweep further. Source: "
         f"`logs/atom/{a.sweep.name}/`, detail in `kimi-k3-maxseqs512.md`.", "",
         "| Concurrency | Run C (cap 256) | **Run D (cap 512)** | D / C | D TTFT med (ms) | D TPOT med (ms) |",
         "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c_ref = RUN_C.get(r["c"])
        ratio = f"**{r['tps']/c_ref:.2f}×**" if c_ref else "—"
        star = "**" if r is best else ""
        L.append(f"| {r['c']} | {f(c_ref) if c_ref else '—'} | {star}{f(r['tps'])}{star} | "
                 f"{ratio} | {f(r['ttft'])} | {f(r['tpot'],2)} |")
    L += [""]

    # Did it keep climbing, or flatten? Judge on the top two concurrency points measured,
    # whatever they are -- filtering on an absolute threshold silently produces no verdict
    # if the sweep is short.
    verdict_lines = []
    if len(rows) >= 2:
        a2, b2 = rows[-2], rows[-1]
        gain = (b2["tps"] - a2["tps"]) / a2["tps"]
        ratio_c = b2["c"] / a2["c"]
        if gain >= 0.20:
            verdict_lines = [
                f"**Still climbing.** c={a2['c']} → {b2['c']} gained **{gain*100:.0f}%** "
                f"({f(a2['tps'])} → {f(b2['tps'])} tok/s). The admission cap was still the "
                f"binding limit at 256; the ceiling has *still* not been found. Raising it "
                f"further is worth another round, though TPOT ({f(b2['tpot'],2)} ms) is now "
                f"the thing to watch — at some point the latency cost stops being acceptable "
                f"even if throughput rises."]
        elif gain <= 0.05:
            verdict_lines = [
                f"**Flattened — the cap is no longer the limit.** c={a2['c']} → {b2['c']} "
                f"gained only **{gain*100:.0f}%** ({f(a2['tps'])} → {f(b2['tps'])} tok/s) "
                f"despite {ratio_c:.0f}× the concurrency. Something other than `max-num-seqs` "
                f"now binds. This is the point where **next-step #2 (profile a step)** stops "
                f"being optional: the residual analysis in §3 is inference, and identifying "
                f"the new limiter needs a real trace, not another sweep."]
        else:
            verdict_lines = [
                f"**Diminishing returns.** c={a2['c']} → {b2['c']} gained "
                f"**{gain*100:.0f}%** ({f(a2['tps'])} → {f(b2['tps'])} tok/s) — real but "
                f"sub-linear. The cap is partially relieved; a second constraint is emerging. "
                f"Profiling (next-step #2) would say which."]
    L += verdict_lines + [""]

    wg, nexp = weight_gb(best["c"])
    step_ms = best["c"] / best["tps"] * 1000
    hbm_pct = 100 * (wg / (step_ms / 1000)) / HBM
    L += [f"At the peak point (c={best['c']}): **{nexp:.0f} of {E} experts** fire per layer, "
          f"{f(wg)} GB/GPU/step of weight traffic, step {step_ms:.1f} ms → HBM "
          f"**~{hbm_pct:.0f}%**. As predicted in §4, HBM utilization keeps falling as batch "
          f"grows while throughput rises — the two move in opposite directions, which is why "
          f"HBM% is a diagnostic and not a target.", "",
          END, ""]
    block = "\n".join(L)

    text = a.md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", block,
                      text, flags=re.DOTALL)
    else:
        anchor = re.search(r"\n## 4\. Next steps", text)
        if anchor:
            text = text[:anchor.start()] + "\n" + block + text[anchor.start():]
        else:
            text = text.rstrip() + "\n\n" + block

    # Mark next-step #1 as done rather than leaving a stale instruction.
    text = text.replace(
        "1. **Extend Run C: `--max-num-seqs 512`, sweep to c=512.** Highest value.",
        "1. ~~**Extend Run C: `--max-num-seqs 512`, sweep to c=512.**~~ **DONE — see Run D "
        "above.** Was highest value.")
    a.md.write_text(text)
    print(f"updated {a.md} with Run D ({len(rows)} points, peak {best['tps']:.1f} tok/s)")


if __name__ == "__main__":
    main()
