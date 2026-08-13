#!/usr/bin/env python3
"""Fold the fp4 scaling investigation's verdict into results/rvs_tflops.md.

Usage: update_rvs_summary_with_investigation.py <fp4_investigation.md> <rvs_tflops.md>

Extracts the per-N consistency verdict from the investigation report and inserts (or
replaces, idempotently, via HTML comment markers) a short section into rvs_tflops.md right
after the "### Why `fp4` alone gains ...x" section that this investigation follows up on.
Safe to re-run: re-running analyze_rvs.py regenerates rvs_tflops.md from scratch (which
would drop this section), so this script must be re-run after analyze_rvs.py, not before.
"""
import argparse, re, sys
from pathlib import Path

BEGIN = "<!-- BEGIN fp4-investigation-result (auto-generated, do not edit by hand) -->"
END = "<!-- END fp4-investigation-result -->"

VERDICT_RE = re.compile(
    r"^- gpu-ids in the bottom half in EVERY repeat: (\{[^}]*\}) \((\d+)% overlap\) -> "
    r"\*\*(.+?)\*\*", re.MULTILINE)
N_HEADER_RE = re.compile(r"^### N=(\d+)$", re.MULTILINE)


def extract_verdicts(investigation_md: str):
    """Return [(N, common_gpus, overlap_pct, verdict), ...] in N order."""
    sections = N_HEADER_RE.split(investigation_md)[1:]  # [N, body, N, body, ...]
    out = []
    for i in range(0, len(sections), 2):
        n, body = sections[i], sections[i + 1]
        m = VERDICT_RE.search(body)
        if m:
            out.append((int(n), m.group(1), int(m.group(2)), m.group(3)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("investigation_md", type=Path)
    ap.add_argument("rvs_tflops_md", type=Path)
    a = ap.parse_args()

    if not a.investigation_md.exists():
        sys.exit(f"not found: {a.investigation_md}")
    if not a.rvs_tflops_md.exists():
        sys.exit(f"not found: {a.rvs_tflops_md}")

    inv_text = a.investigation_md.read_text()
    verdicts = extract_verdicts(inv_text)
    if not verdicts:
        sys.exit("no per-N verdicts found in investigation report — nothing to fold in")

    deterministic = [v for v in verdicts if "consistent" in v[3] and "inconsistent" not in v[3]]
    block = [BEGIN, "",
             "### fp4 N>=5 scaling investigation — result", "",
             f"Follow-up to the finding above (§A.7 of `plan.md`): 3 repeats each at "
             f"N=5,6,7,8, with concurrent `rocm-smi` clock/power sampling. "
             f"Full detail: `results/fp4_investigation.md`.", "",
             "| N | GPU-ids low in every repeat | Overlap | Verdict |",
             "|---:|---|---:|---|"]
    for n, gpus, pct, verdict in verdicts:
        block.append(f"| {n} | {gpus} | {pct}% | {verdict} |")

    block += ["",
              f"**Bottom line**: {len(deterministic)}/{len(verdicts)} tested GPU-counts "
              f"showed a consistent (same-GPU-every-repeat) low performer. " +
              ("This leans toward a real, repeatable per-die effect (hardware/topology-"
               "correlated) rather than a one-off measurement artifact — see the clock/power "
               "columns in `results/fp4_investigation.md` for whether it correlates with "
               "depressed clocks (thermal/power cause) or not (points elsewhere)."
               if deterministic else
               "None of the tested GPU-counts showed a consistent low performer across "
               "repeats — the low performer moved between runs, which points at "
               "non-determinism in RVS's parallel `gst` launch or the fp4 kernel path under "
               "concurrent multi-GPU load, not a specific die."),
              "", END, ""]
    block_text = "\n".join(block)

    text = a.rvs_tflops_md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?",
                       block_text, text, flags=re.DOTALL)
    else:
        marker = re.search(r"(### Why `fp4` alone gains.*?\n\n)(?=##|\Z)", text, re.DOTALL)
        if marker:
            text = text[:marker.end()] + block_text + "\n" + text[marker.end():]
        else:
            text = text.rstrip() + "\n\n" + block_text

    a.rvs_tflops_md.write_text(text)
    print(f"updated {a.rvs_tflops_md} with {len(verdicts)} N-level verdicts "
          f"({len(deterministic)} consistent)")


if __name__ == "__main__":
    main()
