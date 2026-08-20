#!/usr/bin/env python3
"""Fold the single-stream kernel-path result into results/kimi-k3-improve.md.

Usage: update_improve_with_ss.py <kimi_single_stream_dir> <kimi-k3-improve.md>

Idempotent: the block is delimited by HTML comment markers and replaced on re-run.
Reports PER-REQUEST tok/s (1000/TPOT) as primary, matching the section it lands in.
K1_mad_default is the control; arms are compared to it, never to Run A's 46.6 tok/s
(different image -- that comparison would confound image with kernel path).
"""
import argparse, json, re, sys
from pathlib import Path

BEGIN = "<!-- BEGIN single-stream (auto-generated) -->"
END = "<!-- END single-stream -->"
CONTROL = "K1_mad_default"
RUN_A_C1 = 46.6

ARM_DESC = {
    "K1_mad_default":  "MAD baseline (control)",
    "K2_triton_moe":   "`ATOM_USE_TRITON_MOE=1` — MoE kernel path",
    "K3_aiter_attn":   "`ATOM_USE_UNIFIED_ATTN=0` + `ATOM_FORCE_ATTN_TRITON=0` — attention path",
    "K4_grouped_gemm": "`ATOM_USE_TRITON_GEMM=0` + `AITER_USE_GROUPED_GEMM=1` — GEMM path",
}


def load_arm(d: Path):
    rows = {}
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        c = int(data.get("max_concurrency") or re.sub(r"\D", "", j.stem) or 0)
        tpot = data.get("median_tpot_ms")
        rows[c] = dict(tpot=tpot, per_req=(1000.0 / tpot) if tpot else None,
                       agg=data.get("output_throughput"))
    return rows


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("md", type=Path)
    a = ap.parse_args()

    arms = {}
    if a.sweep.is_dir():
        for d in sorted(p for p in a.sweep.iterdir() if p.is_dir()):
            r = load_arm(d)
            if r:
                arms[d.name] = r
    if not arms:
        sys.exit(f"no usable results under {a.sweep}")

    concs = sorted({c for r in arms.values() for c in r})
    names = sorted(arms, key=lambda n: (n != CONTROL, n))
    c1 = concs[0]

    L = [BEGIN, "",
         "#### Result — kernel path does / does not move single-stream speed",
         "",
         f"Source: `{a.sweep.name}`, full detail in `kimi-k3-single-stream.md`. "
         f"Per-request tok/s = `1000 / median TPOT`; `{CONTROL}` is the control.",
         "",
         "| Concurrency | " + " | ".join(f"`{n}`" for n in names) + " |",
         "|---:|" + "---:|" * len(names)]
    for c in concs:
        L.append(f"| {c} | " + " | ".join(f(arms[n].get(c, {}).get("per_req")) for n in names) + " |")
    L.append("")

    ctrl = arms.get(CONTROL)
    if ctrl and c1 in ctrl and ctrl[c1]["per_req"]:
        base = ctrl[c1]["per_req"]
        deltas = [(n, arms[n][c1]["per_req"], 100.0 * (arms[n][c1]["per_req"] / base - 1))
                  for n in names if n != CONTROL and arms[n].get(c1, {}).get("per_req")]
        L += [f"Control at c={c1}: **{f(base)} tok/s per request** "
              f"(TPOT {f(ctrl[c1]['tpot'],2)} ms).", ""]
        if deltas:
            best_n, best_v, best_pct = max(deltas, key=lambda x: x[1])
            worst_pct = min(d[2] for d in deltas)
            L += ["| Arm | per-request tok/s | vs control |", "|---|---:|---:|"]
            L += [f"| `{n}` | {f(v)} | {p:+.1f}% |" for n, v, p in deltas]
            L.append("")
            if best_pct >= 5:
                L += [f"**Positive: `{best_n}` gives {best_pct:+.1f}% over the control** "
                      f"({ARM_DESC.get(best_n,'')}). Configuration-level tuning is not "
                      f"exhausted for per-request speed. This is one measurement at one "
                      f"concurrency — repeat before acting on it."]
            elif best_pct <= -5:
                L += [f"**Negative: every arm is at or below the control** (best `{best_n}` "
                      f"at {best_pct:+.1f}%). The MAD kernel set is already the better choice "
                      f"at low batch; these knobs offer no single-stream win."]
            else:
                L += [f"**Null result: the whole spread is within ±5% of the control** "
                      f"({worst_pct:+.1f}% to {best_pct:+.1f}%). This is informative rather "
                      f"than disappointing — it localizes the ~93% non-weight portion of a "
                      f"single-request step to costs no environment variable can reach: "
                      f"kernel launch overhead, the 186 serialized all-reduces (fixed by "
                      f"TP=8 × 93 layers), and the sequential dependency across the 69 KDA "
                      f"layers. **Configuration-level tuning for per-request speed is "
                      f"closed**; further gains require ATOM/AITER kernel work, and the "
                      f"trace re-parse is the way to target it."]
            L.append("")
        L += [f"Run A's {RUN_A_C1} tok/s at c=1 used `rocm/atom-dev:latest`, a different "
              f"image, so it is context rather than a comparison point — `{CONTROL}` is the "
              f"matched control.", ""]

    L += [END, ""]
    block = "\n".join(L)

    text = a.md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", block,
                      text, flags=re.DOTALL)
    else:
        anchor = "### Ranked next experiments"
        if anchor in text:
            text = text.replace(anchor, block + "\n" + anchor, 1)
        else:
            text = text.rstrip() + "\n\n" + block
    a.md.write_text(text)
    print(f"folded single-stream result ({len(arms)} arms) into {a.md}")


if __name__ == "__main__":
    main()
