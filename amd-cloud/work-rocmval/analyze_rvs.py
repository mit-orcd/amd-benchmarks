#!/usr/bin/env python3
"""Aggregate RVS gst sweeps into results/rvs_tflops.{md,csv} + a scaling table.

Usage: analyze_rvs.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>

Reads each run dir's summary.csv (written by run_tflops.sh, columns:
gpus,precision,per_gpu_peak_tflops,aggregate_tflops,gpus_reporting -- already in
TFLOPS); if absent, falls back to re-parsing the raw <n>x_<prec>.log files for
'[GPU:: <id>] ... GFLOPS <v>'.
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

# MI355X vendor peaks (TFLOPS, dense, no sparsity) -- used for % of peak.
PEAK = {"fp4": 10000.0, "fp6": 5000.0, "bf6": 5000.0, "fp8": 5000.0,
        "bf8": 5000.0, "fp16": 2500.0, "bf16": 2500.0, "fp32": 157.3, "fp64": 78.6}
GFLOPS_RE = re.compile(r"GPU::\s*(\d+)\].*?GFLOPS\s+([\d.]+)")


def from_logs(d: Path):
    rows = []
    for log in sorted(d.glob("*x_*.log")):
        m = re.match(r"(\d+)x_(\w+)\.log", log.name)
        if not m:
            continue
        n, prec = int(m.group(1)), m.group(2)
        peaks = defaultdict(float)
        for gid, val in GFLOPS_RE.findall(log.read_text(errors="replace")):
            peaks[gid] = max(peaks[gid], float(val))
        if not peaks:
            continue
        agg = sum(peaks.values()) / 1000.0          # log is GFLOPS
        rows.append({"gpus": n, "precision": prec, "aggregate_tflops": agg,
                     "avg_per_gpu_tflops": agg / len(peaks),
                     "gpus_reporting": len(peaks)})
    return rows


def load(d: Path):
    csvf = d / "summary.csv"
    if csvf.exists():
        with csvf.open() as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        # drop rows the sweep skipped / failed to parse
        return [r for r in rows if r.get("aggregate_tflops") not in (None, "", "0.00")]
    return from_logs(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in a.dirs:
        rows += [{**r, "run": d.name} for r in load(d)]
    if not rows:
        sys.exit("no RVS results found")

    for r in rows:
        r["gpus"] = int(r["gpus"])
        r["aggregate_tflops"] = float(r["aggregate_tflops"])
        r["gpus_reporting"] = int(r.get("gpus_reporting") or r["gpus"])
        r["avg_per_gpu_tflops"] = float(r.get("avg_per_gpu_tflops") or
                                        r["aggregate_tflops"] / max(r["gpus_reporting"], 1))
        p = PEAK.get(r["precision"])
        r["pct_of_peak"] = round(100 * r["avg_per_gpu_tflops"] / p, 1) if p else ""

    fields = ["run", "gpus", "precision", "aggregate_tflops", "avg_per_gpu_tflops",
              "gpus_reporting", "pct_of_peak", "per_gpu_peak_tflops"]
    with (a.out / "rvs_tflops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # markdown: precision x GPU-count matrix of aggregate TFLOPS + scaling efficiency
    precs = sorted({r["precision"] for r in rows},
                   key=lambda p: list(PEAK).index(p) if p in PEAK else 99)
    ns = sorted({r["gpus"] for r in rows})
    idx = {(r["precision"], r["gpus"]): r for r in rows}

    L = ["# RVS `gst` TFLOPS — MI355X x8 (gfx950, ROCm 7.14)", "",
         "## Aggregate TFLOPS", "",
         "| Precision | " + " | ".join(f"N={n}" for n in ns) +
         " | % peak @N=1 | scaling N=8/N=1 |", "|" + "---|" * (len(ns) + 3)]
    for p in precs:
        cells = [f'{idx[(p, n)]["aggregate_tflops"]:.1f}' if (p, n) in idx else "-" for n in ns]
        one, eight = idx.get((p, 1)), idx.get((p, 8))
        pk = f'{one["pct_of_peak"]}%' if one and one["pct_of_peak"] != "" else "-"
        sc = (f'{eight["aggregate_tflops"] / one["aggregate_tflops"] / 8 * 100:.0f}%'
              if one and eight and one["aggregate_tflops"] else "-")
        L.append(f"| {p} | " + " | ".join(cells) + f" | {pk} | {sc} |")

    L += ["", "## Per-GPU average TFLOPS", "",
          "| Precision | " + " | ".join(f"N={n}" for n in ns) + " |",
          "|" + "---|" * (len(ns) + 1)]
    for p in precs:
        L.append(f"| {p} | " + " | ".join(
            f'{idx[(p, n)]["avg_per_gpu_tflops"]:.1f}' if (p, n) in idx else "-"
            for n in ns) + " |")

    L += ["", "Per-GPU value = peak GFLOPS across log intervals (ignores ramp-up); "
              "aggregate = sum of per-GPU peaks. A per-GPU drop as N grows on a "
              "power-dense precision is OAM-tray power capping, not a kernel regression."]
    (a.out / "rvs_tflops.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rvs_tflops.md and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
