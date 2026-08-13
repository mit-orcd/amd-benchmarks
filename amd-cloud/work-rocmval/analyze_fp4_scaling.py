#!/usr/bin/env python3
"""Analyze investigate_fp4_scaling.sh output: is the fp4 N>=5 per-GPU imbalance
thermal/power-correlated (a real hardware effect) or a launch/scheduling artifact
(a software effect)?

Usage: analyze_fp4_scaling.py <investigation_dir> -o <results_dir>

For each (N, repeat) run directory it:
  1. reads the per-GPU peak TFLOPS from summary.csv
  2. reads the concurrent rocm-smi clock/power samples and averages them per GPU
     over that run's wall-clock window
  3. reports the correlation between "this GPU was a low performer" and
     "this GPU had a lower average sclk/power than its siblings"

Two additional checks answer the question this script exists for:
  - Consistency: is the SAME gpu id the low performer across repeats at a given N?
    (Rank-order Spearman-style: how often does the same id appear in the bottom half.)
  - Clock correlation: do low-TFLOPS GPUs also show low average sclk? If yes, the
    imbalance likely has a real clock/power cause. If low-TFLOPS GPUs have normal
    clocks, the deficit is not explained by clocks and points at the measurement /
    launch path instead.
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

SCLK_RE = re.compile(r"GPU\[(\d+)\]\s*:\s*sclk clock level:\s*\S+:\s*\((\d+)Mhz\)")
POWER_RE = re.compile(r"GPU\[(\d+)\]\s*:\s*Current Socket Graphics Package Power \(W\):\s*([\d.]+)")
PEAK_RE = re.compile(r"gpu(\d+)=([\d.]+)")


def read_peaks(run_dir: Path):
    csvf = run_dir / "summary.csv"
    if not csvf.exists():
        return {}
    with csvf.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    # single (N, fp4) row; per_gpu_peak_tflops is "gpuID=val;gpuID=val;..."
    field = rows[-1].get("per_gpu_peak_tflops", "")
    return {gid: float(v) for gid, v in PEAK_RE.findall(field)}


def read_clocks_power(run_dir: Path):
    """Average sclk (MHz) and power (W) per rocm-smi-index-GPU over the sample window.

    NOTE: rocm-smi indexes GPUs 0..7 by PCI enumeration order, while RVS/gst report the
    driver's internal gpu id (the number in `[GPU:: <id>]`). These are DIFFERENT id
    spaces. We cannot directly join them without a id<->index map, so this function
    returns index-keyed clock/power series, and the caller reports them as an
    unlabeled distribution (min/max/spread) for sanity-checking rather than claiming a
    specific gpu-id-to-clock correlation. See the printed caveat in main().
    """
    log = run_dir / "clocks_power.log"
    if not log.exists():
        return {}, {}
    sclk, power = defaultdict(list), defaultdict(list)
    for line in log.read_text(errors="replace").splitlines():
        m = SCLK_RE.search(line)
        if m:
            sclk[m.group(1)].append(int(m.group(2)))
            continue
        m = POWER_RE.search(line)
        if m:
            power[m.group(1)].append(float(m.group(2)))
    avg_sclk = {k: sum(v) / len(v) for k, v in sclk.items() if v}
    avg_power = {k: sum(v) / len(v) for k, v in power.items() if v}
    return avg_sclk, avg_power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    runs = sorted(p for p in a.dir.glob("n*_r*") if p.is_dir())
    if not runs:
        sys.exit(f"no n<N>_r<repeat> run directories found under {a.dir}")

    L = ["# fp4 N>=5 scaling investigation", "",
         f"Source: `{a.dir}`", "",
         "Per repeat: per-GPU peak TFLOPS (RVS gpu-id space) alongside the concurrently "
         "sampled clock/power distribution (rocm-smi index space -- these two id spaces "
         "are NOT the same GPU numbering and are not directly joined here; the clock/power "
         "columns are a same-run sanity check, not a per-GPU-id correlation).", "",
         "| N | repeat | TFLOPS spread (min-max, GPU-id space) | sclk spread MHz (rocm-smi index space) | power spread W (rocm-smi index space) |",
         "|---|---:|---|---|---|"]

    by_n = defaultdict(list)   # N -> list of {gpu_id: tflops} across repeats
    for run_dir in runs:
        m = re.match(r"n(\d+)_r(\d+)", run_dir.name)
        if not m:
            continue
        n, rep = int(m.group(1)), int(m.group(2))
        peaks = read_peaks(run_dir)
        if not peaks:
            L.append(f"| {n} | {rep} | _no data_ | - | - |")
            continue
        by_n[n].append(peaks)
        sclk, power = read_clocks_power(run_dir)
        tv = list(peaks.values())
        sv, pv = list(sclk.values()), list(power.values())
        L.append(f"| {n} | {rep} | {min(tv):.0f}-{max(tv):.0f} "
                 f"(spread {(max(tv)-min(tv))/max(tv)*100:.0f}%) | "
                 f"{f'{min(sv):.0f}-{max(sv):.0f}' if sv else '-'} | "
                 f"{f'{min(pv):.0f}-{max(pv):.0f}' if pv else '-'} |")

    L += ["", "## Consistency across repeats: is it the same GPU every time?", "",
          "For each N, which gpu-id landed in the bottom half of per-GPU TFLOPS, per repeat. "
          "If the same id(s) appear across all repeats at a given N, that is a deterministic, "
          "likely hardware- or topology-correlated effect. If the low performer changes "
          "between repeats, that points at non-determinism in the launch/sync path instead.",
          ""]
    for n in sorted(by_n):
        repeats = by_n[n]
        L.append(f"### N={n}")
        bottoms = []
        for i, peaks in enumerate(repeats, 1):
            half = max(1, len(peaks) // 2)
            bottom = sorted(peaks, key=peaks.get)[:half]
            bottoms.append(set(bottom))
            L.append(f"- repeat {i}: bottom half = {{{', '.join(sorted(bottom))}}}")
        if len(bottoms) > 1:
            common = set.intersection(*bottoms)
            union = set.union(*bottoms)
            consistency = len(common) / len(union) * 100 if union else 0
            verdict = ("**consistent — likely deterministic/hardware-correlated**"
                       if consistency >= 50 else
                       "**inconsistent — likely non-deterministic/software-correlated**")
            L.append(f"- gpu-ids in the bottom half in EVERY repeat: "
                     f"{{{', '.join(sorted(common)) or 'none'}}} "
                     f"({consistency:.0f}% overlap) -> {verdict}")
        L.append("")

    L += ["## How to read this", "",
          "- **High consistency + low performers also show depressed clocks/power**: a real "
          "per-die thermal or power effect (e.g. VRM zone, cooling asymmetry). Not a bug.",
          "- **High consistency + clocks look normal across the board**: a deterministic "
          "effect not explained by clocks — worth checking topology (NUMA/XGMI placement of "
          "those specific dies) rather than power.",
          "- **Low consistency (different GPU low each repeat)**: points at non-determinism "
          "in RVS's parallel gst launch or in the fp4 MXFP4 kernel path under concurrent "
          "multi-GPU load — a software/scheduling issue, not a hardware one.",
          "", "## Caveat", "",
          "rocm-smi's GPU index and RVS's internal gpu id are different numbering schemes "
          "and are not joined in this analysis (see the docstring). A future pass could "
          "resolve this via `rvs -g` output order, at which point the clock/power columns "
          "could be attributed to specific gpu ids rather than reported as a same-run range.",
          ""]

    (a.out / "fp4_investigation.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/fp4_investigation.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
