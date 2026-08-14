#!/usr/bin/env python3
"""Append the Kimi-K3 EP-vs-TP-only A/B result to results/kimi-k3.md.

Usage: analyze_kimi_ep.py <ep_sweep_dir> <baseline_sweep_dir> <kimi-k3.md>

Idempotent: the block is delimited by HTML comment markers and replaced on re-run.
Interprets the result against the bandwidth hypothesis in section 3/5 rather than just
dumping numbers -- EP is predicted to help because it trades HBM traffic (~29% utilized)
for all-to-all over an idle interconnect (~1% utilized).
"""
import argparse, json, re, sys
from pathlib import Path

BEGIN = "<!-- BEGIN kimi-ep-ab (auto-generated) -->"
END = "<!-- END kimi-ep-ab -->"

FIELDS = ["max_concurrency", "output_throughput", "median_ttft_ms", "median_tpot_ms",
          "completed", "request_throughput"]


def load(d: Path):
    rows = {}
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        c = data.get("max_concurrency") or int(re.sub(r"\D", "", j.stem) or 0)
        rows[int(c)] = {k: data.get(k) for k in FIELDS}
    return rows


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep_dir", type=Path)
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("md", type=Path)
    a = ap.parse_args()

    ep, base = load(a.ep_dir), load(a.base_dir)
    if not ep:
        sys.exit(f"no EP results in {a.ep_dir}")
    if not base:
        sys.exit(f"no baseline results in {a.base_dir}")

    ns = sorted(set(ep) & set(base))
    L = [BEGIN, "",
         "## 6. Experiment — expert parallelism (EP) on vs off", "",
         "Section 5 flagged EP as the top tuning lever: with EP off, TP shards every expert "
         "across all 8 GPUs so each GPU reads a slice of *every* activated expert "
         "(~116 GB/step, ~29% of HBM bandwidth), while XGMI sits ~1% utilized. EP places "
         "whole experts on specific ranks — fewer, complete reads per GPU, paid for with "
         "all-to-all token routing over the idle interconnect.", "",
         "Both runs use the identical validated recipe flag set; the **only** difference is "
         "`--enable-expert-parallel`. Baseline is the reference result (EP is not part of "
         "the validated recipe).", "",
         f"- Baseline (TP-only): `{a.base_dir.name}`",
         f"- EP enabled: `{a.ep_dir.name}`", "",
         "| Concurrency | tok/s TP-only | tok/s EP | EP/TP | TPOT TP-only (ms) | TPOT EP (ms) | TTFT TP-only (ms) | TTFT EP (ms) |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]

    ratios = []
    for n in ns:
        b, e = base[n], ep[n]
        bt, et = b.get("output_throughput"), e.get("output_throughput")
        r = (et / bt) if (bt and et) else None
        if r:
            ratios.append((n, r))
        mark = "**" if r and (r >= 1.10 or r <= 0.90) else ""
        L.append(f"| {n} | {f(bt)} | {f(et)} | {mark}{f(r,2) if r else '-'}x{mark} | "
                 f"{f(b.get('median_tpot_ms'),2)} | {f(e.get('median_tpot_ms'),2)} | "
                 f"{f(b.get('median_ttft_ms'))} | {f(e.get('median_ttft_ms'))} |")

    L += [""]
    if ratios:
        top_n, top_r = max(ratios, key=lambda t: t[0])   # highest concurrency point
        best = max(ratios, key=lambda t: t[1])
        worst = min(ratios, key=lambda t: t[1])
        L += [f"**Headline (c={top_n}, the batch where weight traffic dominates): "
              f"{top_r:.2f}x**  ·  range {worst[1]:.2f}x (c={worst[0]}) to "
              f"{best[1]:.2f}x (c={best[0]}).", ""]

        if top_r >= 1.10:
            verdict = (
                f"**EP helps — hypothesis supported.** At c={top_n}, throughput improves "
                f"{(top_r-1)*100:.0f}%. This is consistent with the section 3 diagnosis: the "
                f"workload was HBM-bandwidth-bound, and moving expert traffic off HBM and "
                f"onto the near-idle interconnect relieved the actual constraint. The gain "
                f"should be largest at high concurrency, where the most distinct experts "
                f"activate per step — check that shape in the table above; if the ratio is "
                f"flat across concurrency, the mechanism is probably *not* the one proposed "
                f"and the win is coming from somewhere else.")
        elif top_r <= 0.90:
            verdict = (
                f"**EP hurts — hypothesis not supported.** At c={top_n}, throughput drops "
                f"{(1-top_r)*100:.0f}%. Bandwidth headroom alone did not predict the outcome. "
                f"The likely reason is that all-to-all is *latency*-sensitive in a way bulk "
                f"all-reduce is not: it is on the critical path twice per MoE layer "
                f"(dispatch + combine) with a synchronization barrier each time, so 92 MoE "
                f"layers x 2 serialized round-trips can cost more wall-clock than the HBM "
                f"traffic it saves — even with ~99% of interconnect bandwidth spare. "
                f"Bandwidth utilization is the wrong metric for a latency-bound collective.")
        else:
            verdict = (
                f"**Roughly neutral ({top_r:.2f}x at c={top_n}).** The HBM saving and the "
                f"added all-to-all latency approximately cancel. That is itself informative: "
                f"it means the workload is not *purely* bandwidth-bound the way section 3's "
                f"utilization figures imply, and that all-to-all round-trip latency across "
                f"92 MoE layers is a first-order cost, not a rounding error.")
        L += [verdict, ""]

        L += ["**Caveat on interpretation.** Bandwidth *utilization* (29% HBM vs 1% XGMI) "
              "motivated this experiment, but utilization measures throughput headroom, not "
              "latency exposure. A collective can be far from bandwidth-saturated and still "
              "dominate step time if it serializes. Whatever the sign of the result above, "
              "the honest lesson is that the two resources are not interchangeable just "
              "because one has spare capacity.", ""]

    L += ["Raw data: per-concurrency JSON in the two sweep directories listed above.", "",
          END, ""]
    block = "\n".join(L)

    text = a.md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", block,
                      text, flags=re.DOTALL)
    else:
        marker = re.search(r"\n---\n\n## Source data", text)
        if marker:
            text = text[:marker.start()] + "\n---\n\n" + block + text[marker.start():]
        else:
            text = text.rstrip() + "\n\n---\n\n" + block
    a.md.write_text(text)
    print(f"appended EP A/B ({len(ns)} concurrency points) to {a.md}")


if __name__ == "__main__":
    main()
