#!/usr/bin/env python3
"""Plot busbw vs N per collective from results/rccl.csv -> results/rccl_busbw.png,
highlighting the non-power-of-2 (N=5,6,7) cliff.

Reads the CSV rather than hardcoding numbers, so the figure regenerates automatically
after every rerun (run_part_b.sh calls this as its last stage).

Color design: 9 collectives is one past the 8-hue categorical ceiling, so instead of a
9th generated hue this uses composite encoding -- 2 categorical hues (palette slots 1
and 2) for the two mechanisms that behave differently at the cliff, and a distinct
marker per collective for identity within each family. This is also the actual finding,
not just a color-count workaround: ring/tree-based collectives collapse ~4-5x at
N=5-7 (a real ring can only be built at power-of-2 GPU counts on this K8 xGMI mesh),
while the three pairwise-sendrecv collectives barely dip, because they route directly
to/from a root rank without needing a closed ring.
"""
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/rccl.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/rccl_busbw.png"

# Reference palette slots 1 (blue) and 2 (orange) -- see dataviz skill palette.md.
# Adjacent-pair CVD/contrast already validated for this exact order (worst adjacent
# CVD dE 9.1 light / 8.4 dark, >= 8 target); using only 2 of the 8 slots here.
RING_COLOR = "#2a78d6"       # blue -- ring/tree-based, closes a ring across the mesh
PAIRWISE_COLOR = "#eb6834"   # orange -- pairwise sendrecv, routes to/from a root
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
CLIFF_BAND = "#f0efec"       # neutral gray -- diverging-pair midpoint tone, reused
                              # here for the shaded cliff band since it must read as
                              # "not a data color", the same rule that governs
                              # the diverging midpoint.

RING_BASED = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce", "alltoall"]
PAIRWISE = ["gather", "scatter", "sendrecv"]
MARKERS = {  # distinct marker shape per collective = identity within a hue family
    "all_reduce": "o", "all_gather": "s", "reduce_scatter": "^",
    "broadcast": "D", "reduce": "v", "alltoall": "P",
    "gather": "o", "scatter": "s", "sendrecv": "^",
}

series = defaultdict(dict)
with open(src) as f:
    for r in csv.DictReader(f):
        if r["config"] != "default":
            continue
        series[r["collective"]][int(r["gpus"])] = float(r["busbw_at_max_GBps"])

if not series:
    sys.exit(f"no default-config rows in {src}")

fig, ax = plt.subplots(figsize=(10, 6))

# Cliff band first, so lines draw on top of it.
all_ns = sorted({n for pts in series.values() for n in pts})
if {5, 6, 7} & set(all_ns):
    ax.axvspan(4.5, 7.5, color=CLIFF_BAND, zorder=0)
    ymax_hint = max(v for pts in series.values() for v in pts.values())
    ax.text(6, ymax_hint * 1.02, "non-power-of-2 cliff",
            ha="center", va="bottom", fontsize=9.5, color=MUTED_INK, style="italic")

for coll in RING_BASED:
    if coll not in series:
        continue
    pts = series[coll]
    ns = sorted(pts)
    ax.plot(ns, [pts[n] for n in ns], marker=MARKERS[coll], markersize=6,
            linewidth=1.8, color=RING_COLOR, alpha=0.85, label=coll)
for coll in PAIRWISE:
    if coll not in series:
        continue
    pts = series[coll]
    ns = sorted(pts)
    ax.plot(ns, [pts[n] for n in ns], marker=MARKERS[coll], markersize=6,
            linewidth=1.8, color=PAIRWISE_COLOR, alpha=0.85, label=coll)

# Ring-based drop magnitude, called out directly rather than left for the reader to
# compute -- the number the chart exists to communicate.
if "all_reduce" in series and 4 in series["all_reduce"] and 5 in series["all_reduce"]:
    v4, v5 = series["all_reduce"][4], series["all_reduce"][5]
    ax.annotate(f"all_reduce: {v4:.0f} -> {v5:.0f} GB/s\n({v4 / v5:.1f}x drop)",
                xy=(5, v5), xytext=(5.4, v5 + (ymax_hint * 0.10)),
                fontsize=8.5, color=MUTED_INK,
                arrowprops=dict(arrowstyle="-", color=MUTED_INK, lw=0.8))

for n in (2, 4, 8):
    ax.axvline(n, color=GRID_COLOR, lw=1, zorder=0)

ax.set_xlabel("GPUs (N)", color=PRIMARY_INK)
ax.set_ylabel("busbw (GB/s)", color=PRIMARY_INK)
ax.set_title("RCCL busbw at top message size — MI355X x8, XGMI (AMD Cloud)\n"
             "Ring-based collectives collapse at N=5–7; pairwise (gather/scatter/sendrecv) barely dip",
             color=PRIMARY_INK, fontsize=11)
ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color(AXIS_COLOR)
ax.tick_params(colors=MUTED_INK)

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, ncol=2, fontsize=8.5, frameon=False,
                 title="blue = ring-based   orange = pairwise", title_fontsize=8.5,
                 loc="upper left")
leg.get_title().set_color(MUTED_INK)

fig.tight_layout()
fig.savefig(dst, dpi=150, facecolor="white")
print("wrote", dst)
