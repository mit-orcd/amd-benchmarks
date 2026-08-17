#!/usr/bin/env python3
"""Insert the AMD Cloud Megatron-LM reference result into PRIMUS_REPORT.md section 1.2,
turning the two-column B200-vs-Dell table into a genuine three-way comparison.

Usage: update_b200_table.py <STATE.txt> <PRIMUS_REPORT.md>

Idempotent via HTML comment markers. Reads TFLOPS= and N= from the run's STATE file.
"""
import re, sys
from pathlib import Path

BEGIN = "<!-- BEGIN megatron-ref-3way (auto-generated) -->"
END = "<!-- END megatron-ref-3way -->"

B200 = 986.0
DELL = 790.4


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: update_b200_table.py <STATE.txt> <report.md>")
    state, md = Path(sys.argv[1]), Path(sys.argv[2])
    txt = state.read_text()
    m = re.search(r"^TFLOPS=([\d.]+)$", txt, re.M)
    n = re.search(r"^N=(\d+)$", txt, re.M)
    if not m:
        sys.exit("no TFLOPS= in STATE file — nothing to insert")
    ours = float(m.group(1))
    N = int(n.group(1)) if n else 8

    L = [BEGIN, "",
         f"#### 1.2a Three-way, matched workload (N={N})", "",
         "This row **is** apples-to-apples. It reproduces the exact Dell Cloud configuration "
         "on this host — same GPT-15.6B shape (L=40, H=6144, FFN=16384, heads=48, GQA kv=8, "
         "seq=4096, vocab=50304), same MBS=4 / GBS=32, BF16, no recompute, TP=PP=1, "
         "distributed optimizer, `--ddp-bucket-size 250000000`, 50 iters — using the "
         "**ROCm/Megatron-LM image, not Primus**, so there is no primus-turbo advantage. "
         "**`HSA_OVERRIDE_GFX_VERSION=9.4.2` is set on both sides** — required here too "
         "(the image's torch has no compiled gfx950 code objects at all; unset, backward "
         "pass fails in Transformer Engine with `RuntimeError: Unable to find any suitable "
         "algorithms`), so this comparison is same config *and* same code objects, not just "
         "same config.", "",
         "| Machine | TF/s/GPU | vs B200 | vs Dell MI355X |",
         "|---|---:|---:|---:|",
         f"| NVIDIA B200 | {B200:.1f} | 100% | — |",
         f"| Dell Cloud MI355X | {DELL:.1f} | {100*DELL/B200:.1f}% | 1.00x |",
         f"| **AMD Cloud MI355X** | **{ours:.1f}** | **{100*ours/B200:.1f}%** | **{ours/DELL:.2f}x** |",
         ""]

    delta = ours / DELL
    if delta >= 1.03:
        L += [f"AMD Cloud comes out **{(delta-1)*100:.0f}% ahead of Dell Cloud** on identical "
              f"workload, image family, *and* code objects (both run gfx942 kernels via the "
              f"override — see above). So this is not the gfx950-vs-gfx942 effect seen "
              f"elsewhere in this report (e.g. RVS fp4, plan.md §A). Candidates here are "
              f"newer ROCm/driver/amdgpu on this host, or run-to-run variance on identical "
              f"software. This also moves the MI355X-vs-B200 ratio from "
              f"{100*DELL/B200:.1f}% to **{100*ours/B200:.1f}%**.", ""]
    elif delta <= 0.97:
        L += [f"AMD Cloud is **{(1-delta)*100:.0f}% behind Dell Cloud** despite identical "
              f"configuration. Since the silicon is the same, this points at the software "
              f"stack or image version rather than hardware — worth investigating before "
              f"treating either number as definitive.", ""]
    else:
        L += [f"The two MI355X hosts agree within {abs(delta-1)*100:.0f}% on identical "
              f"workload — a clean cross-machine validation. The MI355X-vs-B200 gap is "
              f"therefore a property of the hardware/software stack, not of either "
              f"particular machine.", ""]

    L += ["> **What makes this different from section 1.1.** Section 1.1 is llama2-7B via "
          "Primus with primus-turbo — a different model, framework, and kernel set, and "
          "*not* comparable to the B200 number. This section deliberately abandons Primus "
          "to match the reference configuration exactly. Both are valid; they answer "
          "different questions.", "",
          END, ""]
    block = "\n".join(L)

    text = md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", block,
                      text, flags=re.DOTALL)
    else:
        anchor = re.search(r"\n## 2\. GEMM microbench", text)
        if anchor:
            text = text[:anchor.start()] + "\n" + block + text[anchor.start():]
        else:
            text = text.rstrip() + "\n\n" + block
    md.write_text(text)
    print(f"inserted 3-way table: AMD={ours:.1f} Dell={DELL} B200={B200}")


if __name__ == "__main__":
    main()
