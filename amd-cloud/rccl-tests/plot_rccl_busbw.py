#!/usr/bin/env python3
"""Plot busbw vs N per collective from results/rccl.csv -> results/rccl_busbw.png

Unlike the reference version (which hardcoded its numbers in a `data = {...}` dict),
this reads the CSV, so the figure regenerates automatically after every rerun.
"""
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/rccl.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/rccl_busbw.png"

series = defaultdict(dict)
with open(src) as f:
    for r in csv.DictReader(f):
        if r["config"] != "default":
            continue
        series[r["collective"]][int(r["gpus"])] = float(r["busbw_at_max_GBps"])

if not series:
    sys.exit(f"no default-config rows in {src}")

fig, ax = plt.subplots(figsize=(9, 5.5))
for coll, pts in sorted(series.items()):
    ns = sorted(pts)
    ax.plot(ns, [pts[n] for n in ns], marker="o", label=coll)
for n in (2, 4, 8):
    ax.axvline(n, color="0.85", lw=1, zorder=0)
ax.set(xlabel="GPUs (N)", ylabel="busbw (GB/s)",
       title="RCCL busbw at top message size — MI355X x8, XGMI")
ax.grid(alpha=.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(dst, dpi=150)
print("wrote", dst)
