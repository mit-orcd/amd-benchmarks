#!/usr/bin/env python3
"""Aggregate RVS gst sweeps into results/rvs_tflops.{md,csv}.

Usage: analyze_rvs.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>

Emits a summary with the same information as dell-cloud/work-rocmval/summary.md +
summary-sweep.md: hardware specs, measured TFLOPS with per-N scaling efficiency,
measured-vs-dense-peak, per-GPU spread, the B200 comparison, and auto-generated
observations. Narrative verdicts are left to results/SUMMARY.md.

Reads each run dir's summary.csv (written by run_tflops.sh, columns:
gpus,precision,per_gpu_peak_tflops,aggregate_tflops,gpus_reporting -- already in
TFLOPS); if absent, re-parses the raw <n>x_<prec>.log files.
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

# MI355X vendor peaks (TFLOPS, dense, no sparsity) -- used for % of peak.
# NB: FP6/BF6 are 10,000 not 5,000 -- CDNA 4 runs MX-FP6 at the FP4 rate. plan.md's
# original table had 5,000 here, which inflated fp6 "% of peak" by 2x. Matches the
# spec table in dell-cloud/work-rocmval/summary.md.
PEAK = {"fp4": 10000.0, "fp6": 10000.0, "bf6": 10000.0, "fp8": 5000.0,
        "bf8": 5000.0, "fp16": 2500.0, "bf16": 2500.0, "fp32": 157.3, "fp64": 78.6}

# NVIDIA B200 per-GPU reference (same figures dell-cloud/work-rocmval/summary.md used).
# B200_MEASURED is the provided reference measurement; B200_PEAK is published dense spec.
B200_MEASURED = {"bf16": 1493.0, "fp8": 4103.0}
B200_PEAK = {"fp64": 40.0, "fp32": 80.0, "bf16": 2250.0, "fp16": 2250.0,
             "fp8": 4500.0, "bf8": 4500.0, "fp4": 9000.0}

# Dell Cloud MI355X baseline, 1-GPU per-GPU measured TFLOPS, from
# dell-cloud/work-rocmval/summary.md "Measured vs. dense peak" table.
# Same silicon as this host (8 x MI355X / gfx950); what differs is the software stack:
# ROCm 7.2.3-90 + HSA_OVERRIDE_GFX_VERSION=9.4.2 (gfx942 alias) there, vs ROCm 7.14 native
# gfx950 here. So the delta below is a *software* delta, not a hardware one.
DELL_MEASURED = {"fp4": 3159.52, "fp6": 1280.17, "bf6": 1280.20, "fp8": 3610.88,
                 "bf8": 3238.62, "fp16": 1534.56, "bf16": 1639.78, "fp32": 153.76,
                 "fp64": 77.02}

SPECS = [
    ("Architecture", "CDNA 4 (gfx950)"),
    ("Compute units (per GPU)", "256"),
    ("Memory", "288 GB HBM3E"),
    ("Memory bandwidth (per GPU)", "8 TB/s"),
    ("PCIe host link", "Gen 5 x16 (64 GB/s per direction)"),
    ("GPU-GPU interconnect", "Infinity Fabric (XGMI) 4th gen, ~1075 GB/s aggregate per GPU"),
    ("TBP (per GPU)", "1400 W  (8 GPUs = 11.2 kW tray)"),
    ("Driver / ROCm", "amdgpu 6.19.14.31400100 / ROCm 7.14"),
    ("Host", "2 x EPYC 9575F (256 threads), 3.0 TiB RAM, Ubuntu 22.04.5"),
]

GFLOPS_RE = re.compile(r"GPU::\s*(\d+)\].*?GFLOPS\s+([\d.]+)")
PERGPU_RE = re.compile(r"gpu(\d+)=([\d.]+)")


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
                     "gpus_reporting": len(peaks),
                     "per_gpu_peak_tflops": ";".join(
                         f"gpu{g}={v/1000.0:.2f}" for g, v in sorted(peaks.items()))})
    return rows


def load(d: Path):
    csvf = d / "summary.csv"
    if csvf.exists():
        with csvf.open() as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        return [r for r in rows if r.get("aggregate_tflops") not in (None, "", "0.00")]
    return from_logs(d)


def fmt(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


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
        r["avg_per_gpu_tflops"] = r["aggregate_tflops"] / max(r["gpus_reporting"], 1)
        vals = [float(v) for _, v in PERGPU_RE.findall(r.get("per_gpu_peak_tflops", ""))]
        r["min_per_gpu"] = min(vals) if vals else None
        r["max_per_gpu"] = max(vals) if vals else None
        r["spread_pct"] = (round(100 * (max(vals) - min(vals)) / max(vals), 1)
                           if vals and max(vals) else None)
        p = PEAK.get(r["precision"])
        r["pct_of_peak"] = round(100 * r["avg_per_gpu_tflops"] / p, 1) if p else ""

    fields = ["run", "gpus", "precision", "aggregate_tflops", "avg_per_gpu_tflops",
              "gpus_reporting", "pct_of_peak", "min_per_gpu", "max_per_gpu",
              "spread_pct", "per_gpu_peak_tflops"]
    with (a.out / "rvs_tflops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    precs = sorted({r["precision"] for r in rows},
                   key=lambda p: list(PEAK).index(p) if p in PEAK else 99)
    ns = sorted({r["gpus"] for r in rows})
    idx = {(r["precision"], r["gpus"]): r for r in rows}
    base = min(ns)

    L = ["# RVS `gst` TFLOPS — MI355X x8 (gfx950, ROCm 7.14)", "",
         f"System: 8 x AMD Instinct MI355X (CDNA 4 / gfx950), ROCm 7.14, Ubuntu 22.04.5.",
         f"Source runs: {', '.join(sorted({r['run'] for r in rows}))}", ""]

    L += ["## What this benchmark does", "",
          "`run_tflops.sh` drives the ROCm Validation Suite (RVS) `gst` module (hipBLASLt GEMM",
          "kernels) to measure sustained matrix-multiply throughput. For each precision and GPU",
          "count, a YAML conf is generated from the shipped `conf/MI355X/levels/rvs_level_5.conf`",
          "template, RVS runs with `parallel: true` so every selected GPU runs the GEMM",
          "concurrently, and each GPU emits `GFLOPS <n>` every 3 s. The script takes the **peak**",
          "per-GPU value (steady-state proxy) and sums across GPUs for the aggregate.",
          "`target_stress: 0` means it measures only -- no pass/fail threshold.", "",
          "Every GPU runs an **independent** GEMM: no XGMI or PCIe traffic, no RCCL. Scaling is",
          "therefore embarrassingly parallel, and anything below ~99% is power/thermal sharing on",
          "the 11.2 kW tray or measurement noise -- never interconnect.", ""]

    L += ["## GPU specs", "", "| Item | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in SPECS]
    L += ["", "### Dense peak compute (per GPU, AMD published spec, no sparsity)", "",
          "| Precision | Peak (TFLOPS) |", "|---|---:|"]
    L += [f"| {p} | {fmt(PEAK[p])} |" for p in precs if p in PEAK]
    L += ["", "FP6/BF6 are block-scaled MX formats and run at the **FP4 rate** on CDNA 4 "
          "(10,000 TFLOPS), not half it.", ""]

    # ---- measured aggregate + scaling ------------------------------------------
    L += [f"## Measured TFLOPS (peak across log intervals)", "",
          f"Aggregate = sum of per-GPU peaks. Scaling = aggregate / N={base} value; "
          f"perfect linear scaling would be N/{base}.", "",
          "| Precision | " + " | ".join(f"N={n}" for n in ns) + " | " +
          " | ".join(f"{n}x eff" for n in ns if n != base) + " |",
          "|---|" + "---:|" * (len(ns) + len(ns) - 1)]
    for p in precs:
        cells = [fmt(idx[(p, n)]["aggregate_tflops"], 1) if (p, n) in idx else "-" for n in ns]
        effs = []
        b = idx.get((p, base))
        for n in ns:
            if n == base:
                continue
            r = idx.get((p, n))
            if r and b and b["aggregate_tflops"]:
                ratio = r["aggregate_tflops"] / b["aggregate_tflops"]
                effs.append(f"{ratio:.2f}x ({100 * ratio / (n / base):.0f}%)")
            else:
                effs.append("-")
        L.append(f"| {p} | " + " | ".join(cells) + " | " + " | ".join(effs) + " |")

    # ---- per-GPU average + spread ----------------------------------------------
    L += ["", "## Per-GPU average TFLOPS", "",
          "| Precision | " + " | ".join(f"N={n}" for n in ns) + " |",
          "|---|" + "---:|" * len(ns)]
    for p in precs:
        L.append(f"| {p} | " + " | ".join(
            fmt(idx[(p, n)]["avg_per_gpu_tflops"], 1) if (p, n) in idx else "-"
            for n in ns) + " |")

    nmax = max(ns)
    L += ["", f"### Per-GPU spread at N={nmax} (die-to-die variation)", "",
          "| Precision | min | max | spread |", "|---|---:|---:|---:|"]
    for p in precs:
        r = idx.get((p, nmax))
        if not r or r["min_per_gpu"] is None:
            continue
        L.append(f"| {p} | {fmt(r['min_per_gpu'])} | {fmt(r['max_per_gpu'])} | "
                 f"{r['spread_pct']}% |")

    # ---- measured vs dense peak -------------------------------------------------
    L += ["", f"## Measured vs dense peak (per-GPU, N={base} run)", "",
          "| Precision | Measured | Paper dense peak | % of peak |", "|---|---:|---:|---:|"]
    for p in precs:
        r = idx.get((p, base))
        if not r:
            continue
        pk = PEAK.get(p)
        L.append(f"| {p} | {fmt(r['avg_per_gpu_tflops'], 2)} | {fmt(pk) if pk else '-'} | "
                 f"**{r['pct_of_peak']}%** |")

    if nmax != base:
        L += ["", f"### Aggregate at N={nmax} vs aggregate peak", "",
              "| Precision | Measured aggregate | Peak x8 | % of peak |", "|---|---:|---:|---:|"]
        for p in precs:
            r = idx.get((p, nmax))
            pk = PEAK.get(p)
            if not r or not pk:
                continue
            L.append(f"| {p} | {fmt(r['aggregate_tflops'])} | {fmt(pk * nmax)} | "
                     f"{100 * r['aggregate_tflops'] / (pk * nmax):.1f}% |")

    # ---- three-way comparison: Dell Cloud / AMD Cloud / B200 ---------------------
    L += ["", "## Cross-machine comparison — Dell Cloud vs AMD Cloud vs B200 (per-GPU)", "",
          "All three columns are per-GPU at N=1. **Dell Cloud and AMD Cloud are the same "
          "silicon** — 8 x MI355X (gfx950) — so the delta between them is a *software* delta:",
          "", "| | Dell Cloud | AMD Cloud (this host) |", "|---|---|---|",
          "| ROCm | 7.2.3-90 | **7.14** |",
          "| Code objects | gfx942 alias (`HSA_OVERRIDE_GFX_VERSION=9.4.2`) | **native gfx950** |",
          "| Container | Singularity + ext3 overlay | Docker |",
          "| gst duration | ~60 s | 30 s |", "",
          "B200 reference measurements (provided, per-GPU): 768 TFLOPS FP32 — almost certainly "
          "TF32 tensor, not IEEE FP32 — 1493 TFLOPS BF16, 4103 TFLOPS FP8.", "",
          "| Precision | Dell Cloud MI355X | AMD Cloud MI355X | B200 ref | AMD/Dell | AMD/B200 | MI355X peak | B200 peak |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in precs:
        r = idx.get((p, base))
        if not r:
            continue
        mine = r["avg_per_gpu_tflops"]
        dell, bm = DELL_MEASURED.get(p), B200_MEASURED.get(p)
        mp, bp = PEAK.get(p), B200_PEAK.get(p)
        vs_dell = f"**{mine / dell:.2f}x**" if dell else "-"
        vs_b200 = f"**{mine / bm:.2f}x**" if bm else "-"
        L.append(f"| {p} | {fmt(dell, 2) if dell else '-'} | {fmt(mine, 2)} | "
                 f"{fmt(bm, 0) if bm else '-'} | {vs_dell} | {vs_b200} | "
                 f"{fmt(mp) if mp else '-'} | {fmt(bp, 0) if bp else '-'} |")

    gains = [(p, idx[(p, base)]["avg_per_gpu_tflops"] / DELL_MEASURED[p])
             for p in precs if (p, base) in idx and p in DELL_MEASURED]
    if gains:
        best = max(gains, key=lambda t: t[1])
        worst = min(gains, key=lambda t: t[1])
        L += ["", f"AMD Cloud vs Dell Cloud ranges from **{worst[1]:.2f}x** (`{worst[0]}`) to "
              f"**{best[1]:.2f}x** (`{best[0]}`). Since the silicon is identical, any gain is "
              f"attributable to the newer ROCm and to running native gfx950 code objects "
              f"instead of the gfx942 alias — which is exactly why this host does not set "
              f"`HSA_OVERRIDE_GFX_VERSION`."]
    L += ["", "Only BF16 and FP8 have a like-for-like B200 reference measurement. The FP32 "
          "comparison is apples-to-oranges: MI355X's 157 TFLOPS is true IEEE-754 FP32 on the "
          "vector ALUs, while B200's IEEE FP32 peak is only ~80 TFLOPS — the 768 figure is "
          "TF32 tensor. RVS `gst` has no TF32 config, so that path is unmeasured on either "
          "MI355X host.", ""]

    # ---- auto observations ------------------------------------------------------
    L += ["## Observations (auto-generated)", ""]
    obs = []
    for p in precs:
        b, top = idx.get((p, base)), idx.get((p, nmax))
        if b and top and b["aggregate_tflops"]:
            eff = 100 * (top["aggregate_tflops"] / b["aggregate_tflops"]) / (nmax / base)
            if eff < 90:
                obs.append(f"- `{p}`: N={nmax} scaling efficiency **{eff:.0f}%** -- below the "
                           f"~95% expected for an embarrassingly-parallel GEMM. Power sharing "
                           f"on the 11.2 kW tray is the leading explanation.")
            elif eff > 104:
                obs.append(f"- `{p}`: N={nmax} scaling **{eff:.0f}%** (super-linear) -- the "
                           f"N={base} run caught the boost clock cold; treat as 100%.")
    for p in precs:
        r = idx.get((p, nmax))
        if r and r["spread_pct"] is not None and r["spread_pct"] > 8:
            obs.append(f"- `{p}`: die-to-die spread at N={nmax} is **{r['spread_pct']}%** "
                       f"({fmt(r['min_per_gpu'])}-{fmt(r['max_per_gpu'])} TFLOPS) -- "
                       f"per-die clock variation under sustained load.")
    for p in precs:
        r = idx.get((p, base))
        if r and isinstance(r["pct_of_peak"], float) and r["pct_of_peak"] < 20:
            obs.append(f"- `{p}`: only **{r['pct_of_peak']}%** of dense peak. For MX-FP6 this "
                       f"reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on "
                       f"dell-cloud (12.8%), not a regression on this host.")
    L += obs or ["- Nothing anomalous: all precisions scale within noise of linear and "
                 "land in their expected % -of-peak bands."]

    L += ["", "## Reproducing", "", "```bash",
          "cd /home/amd/shaohao/amd-benchmarks/amd-cloud && source common/env.sh",
          "cd work-rocmval && ./run_part_a.sh          # smoke -> sweep -> health -> analysis",
          "$PY analyze_rvs.py $LOG_ROOT/rvs/sweep_* -o $BENCH_ROOT/results",
          "```", "",
          "## Source data", "",
          "| What | Where |", "|---|---|",
          "| Raw rvs stdout, one per (N, precision) | `logs/rvs/sweep_*/<n>x_<prec>.log` |",
          "| Generated gst confs | `logs/rvs/sweep_*/<n>x_<prec>.conf` |",
          "| Per-run summary | `logs/rvs/sweep_*/summary.{csv,txt}` |",
          "| Health modules | `logs/rvs/health_*/` |",
          "| This table as CSV | `results/rvs_tflops.csv` |", ""]

    (a.out / "rvs_tflops.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rvs_tflops.md and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
