#!/usr/bin/env python3
"""Summarise a captured torch/kineto trace into results/kimi-k3-profile.md.

Usage: analyze_profile.py <trace_dir> <run_out_dir> -o <results_dir>

Aggregates GPU kernel time by name from the trace's `traceEvents`, groups kernels into
coarse classes (MoE/GEMM, attention, collective, memory, other), and reports where step
time actually goes -- replacing the residual-based estimate in kimi-k3-improve.md §3 with
measurement.

Deliberately conservative: if the trace format is not what is expected, it says so and
reports what it could parse rather than inventing a breakdown.
"""
import argparse, gzip, json, re, sys
from collections import defaultdict
from pathlib import Path

# Coarse kernel classes. Order matters -- first match wins.
CLASSES = [
    ("collective",  r"nccl|rccl|all_?reduce|allgather|all_?gather|reduce_scatter|broadcast"),
    ("attention",   r"attn|attention|flash|mla|kda|softmax|rope"),
    ("MoE / GEMM",  r"gemm|moe|expert|matmul|mm_|hipblas|situ|fp4|mxfp4|dequant"),
    ("memory",      r"memcpy|memset|copy_|cast|convert|transpose|reshape|permute"),
    ("norm / act",  r"rmsnorm|layernorm|norm|silu|gelu|activation"),
]


def classify(name: str) -> str:
    n = name.lower()
    for label, pat in CLASSES:
        if re.search(pat, n):
            return label
    return "other"


def load_events(trace_dir: Path):
    """Return (events, source_file). Handles .json and .json.gz."""
    files = sorted(list(trace_dir.rglob("*.json")) + list(trace_dir.rglob("*.json.gz")),
                   key=lambda p: p.stat().st_size, reverse=True)
    for fp in files:
        try:
            if fp.suffix == ".gz":
                with gzip.open(fp, "rt") as fh:
                    data = json.load(fh)
            else:
                data = json.load(fp.open())
        except (OSError, json.JSONDecodeError, EOFError):
            continue
        ev = data.get("traceEvents") if isinstance(data, dict) else None
        if ev:
            return ev, fp
    return None, (files[0] if files else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("run_out", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    files = list(a.trace_dir.rglob("*.json*"))
    total_sz = sum(f.stat().st_size for f in files) / 1e6

    L = ["# Kimi-K3 — profiler trace summary (next-step #2)", "",
         "Replaces the residual-based estimate in `kimi-k3-improve.md` §3 with measured",
         "kernel time. That section attributed ~80% of step time to \"prefill + scheduling\"",
         "by subtracting estimated costs — this is the direct measurement instead.", "",
         f"Traces: `{a.trace_dir}` ({len(files)} file(s), {total_sz:.0f} MB) — kept off-repo,",
         f"driver log `{a.run_out.name}`.", ""]

    events, src = load_events(a.trace_dir)
    if not events:
        L += ["## Trace could not be parsed", "",
              "No `traceEvents` array was found in any captured file. The capture step "
              "reported files on disk, so the profiler ran, but the format is not the "
              "chrome-trace JSON this script expects (it may be a raw kineto/rocprof dump, "
              "or gzipped differently).", "",
              f"Largest candidate file: `{src.name if src else 'none'}`", "",
              "**The raw traces are retained** at the path above — the data is not lost, it "
              "just needs a different reader. ATOM ships `tools/parse_trace.py` and "
              "`tools/analyze_trace_summary.py`, which are format-aware and are the right "
              "next thing to try:", "",
              "```bash",
              f"python ATOM/tools/analyze_trace_summary.py {a.trace_dir}",
              "```", ""]
        (a.out / "kimi-k3-profile.md").write_text("\n".join(L) + "\n")
        print("wrote kimi-k3-profile.md (unparsed-trace report)")
        return

    # Aggregate GPU-side kernel durations.
    by_kernel = defaultdict(float)
    by_class = defaultdict(float)
    n_kern = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_op", "gpu_user_annotation", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur") or 0
        if dur <= 0:
            continue
        name = e.get("name", "?")
        by_kernel[name] += dur
        by_class[classify(name)] += dur
        n_kern += 1

    if not by_kernel:
        L += ["## No GPU kernel events found", "",
              f"Parsed {len(events):,} trace events from `{src.name}` but none carried a GPU "
              "kernel category. The trace may be CPU-only, or the categories differ in this "
              "kineto version. Raw traces retained; try ATOM's own "
              "`tools/analyze_trace_summary.py`.", ""]
        (a.out / "kimi-k3-profile.md").write_text("\n".join(L) + "\n")
        print("wrote kimi-k3-profile.md (no kernel events)")
        return

    total = sum(by_class.values())
    L += [f"Parsed **{n_kern:,} GPU kernel events** from `{src.name}`, "
          f"{total/1e6:.2f} s of total GPU kernel time.", "",
          "## Where GPU time goes, by kernel class", "",
          "| Class | GPU time (ms) | Share |", "|---|---:|---:|"]
    for cls, dur in sorted(by_class.items(), key=lambda kv: -kv[1]):
        L.append(f"| {cls} | {dur/1e3:,.1f} | **{100*dur/total:.1f}%** |")
    L += [""]

    moe = by_class.get("MoE / GEMM", 0) / total * 100
    att = by_class.get("attention", 0) / total * 100
    coll = by_class.get("collective", 0) / total * 100
    L += ["### Reading this against the estimate it replaces", "",
          f"`kimi-k3-improve.md` §3 estimated weight reads at ~4%, compute ~12%, collectives "
          f"~10%, with ~70% unaccounted. Measured here: **MoE/GEMM {moe:.0f}%**, "
          f"**attention {att:.0f}%**, **collectives {coll:.0f}%**.", ""]
    if coll > 20:
        L += [f"**Collectives are far more expensive than the bandwidth estimate implied** "
              f"({coll:.0f}% of GPU time vs ~1-2% bandwidth utilization). That is the "
              f"latency-vs-bandwidth distinction the report flagged: 186 serialized "
              f"all-reduces per token cost real wall-clock even while moving trivial bytes. "
              f"It also means EP -- which would *add* all-to-all -- is even less attractive "
              f"than §4 concluded.", ""]
    elif att > moe:
        L += [f"**Attention dominates over MoE/GEMM** ({att:.0f}% vs {moe:.0f}%), which the "
              f"weight-traffic analysis did not predict. The KDA/MLA hybrid attention path, "
              f"not expert weight reads, is where the step time actually goes at this "
              f"operating point — and that redirects optimization effort entirely.", ""]
    else:
        L += [f"**MoE/GEMM dominates ({moe:.0f}%)**, consistent with the weight-traffic "
              f"picture in §3. The bandwidth-bound diagnosis holds up under measurement.", ""]

    L += ["## Top 25 kernels by total GPU time", "",
          "| Kernel | Class | GPU time (ms) | Share |", "|---|---|---:|---:|"]
    for name, dur in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:25]:
        short = (name[:88] + "…") if len(name) > 88 else name
        L.append(f"| `{short}` | {classify(name)} | {dur/1e3:,.1f} | {100*dur/total:.1f}% |")
    L += ["",
          "## Caveats", "",
          "- **GPU kernel time only.** Time the GPU spends idle waiting on the host "
          "(scheduler, Python, request admission) does not appear here — so this explains "
          "how busy time is spent, not the gap between busy time and wall-clock step time.",
          "- **Kernel classification is regex-based** over kernel names; a mis-named kernel "
          "lands in `other`. Check the top-25 table before trusting a class total.",
          "- **One operating point** (c=256, cap 256). Shares shift with batch size.", "",
          "## Source data", "", "| What | Where |", "|---|---|",
          f"| Raw traces (off-repo) | `{a.trace_dir}` |",
          f"| Capture driver log | `logs/atom/{a.run_out.name}/` |",
          "| Estimate this replaces | `kimi-k3-improve.md` §3 |", ""]

    (a.out / "kimi-k3-profile.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/kimi-k3-profile.md ({n_kern:,} kernels, {len(by_class)} classes)")


if __name__ == "__main__":
    main()
