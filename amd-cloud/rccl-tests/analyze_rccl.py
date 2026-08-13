#!/usr/bin/env python3
"""Parse rccl-tests logs into results/rccl.{md,csv}.

Usage: analyze_rccl.py <log_dir> [<log_dir> ...] -o results

Emits the same information as dell-cloud/rccl-tests/summary-rccl.md: measured busbw
per collective x N, Infinity Fabric paper-spec ceilings with achieved %, the
non-power-of-2 cliff analysis, and the collective/algorithm/knob reference.

Parses raw logs directly (not the summary text), so it works on partial runs.
Handles both naming schemes: <coll>_n<N>.log and <coll>_<config>_n<N>.log
"""
import argparse, csv, re
from pathlib import Path

NAME = re.compile(
    r"^(?P<coll>[a-z_]+?)(?:_(?P<cfg>default|tree|ring|no_mscll|proto_simple))?_n(?P<n>\d+)\.log$")

# MI350-series published Infinity Fabric peaks. Telemetry (amd-smi topology NUMA BW,
# rocm-smi --shownodesbw) reports N/A on this driver build, so these are paper numbers.
LINK_BIDIR = 153.6          # GB/s per xGMI link, bidirectional
LINK_UNIDIR = 76.8          # GB/s per xGMI link, per direction
MAX_LINKS = 7               # each GPU has 7 xGMI links (K8 mesh, every pair 1 hop)

# Which collectives close a ring across the mesh vs route pairwise to/from a root.
RING_COLLS = {"all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce",
              "all_reduce_bias"}
PAIRWISE_COLLS = {"gather", "scatter", "alltoall", "alltoallv", "hypercube"}

# Dell Cloud MI355X baseline, busbw at 8 GiB, from dell-cloud/rccl-tests/summary-rccl.md
# section 1.2 (anchor points only). Same silicon and same fabric as this host -- only the
# software stack differs (ROCm 7.2.3 + gfx942 alias there, ROCm 7.14 native gfx950 here).
DELL_RCCL = {("sendrecv", 2): 59.21, ("all_reduce", 4): 166.48, ("all_reduce", 8): 381.27,
             ("reduce_scatter", 8): 407.69, ("gather", 8): 444.15, ("scatter", 8): 426.40}
DELL_CLIFF_NOTE = "~38 GB/s at N=5/6/7 (~7% of ceiling)"

# Dell Cloud's FULL all-collective sweep, busbw (GB/s) at 8 GiB, from
# dell-cloud/rccl-tests/summary-rccl.md section 1.1. sendrecv was captured there by a
# separate standalone sweep on the same env/config; alltoallv is Dell's own omission (their
# run OOMed at N=5, see their §1.1 note) so it is not compared here either -- our sweep caps
# alltoallv smaller for the same reason (ALLTOALL_MAX), so there is no valid 8 GiB point on
# either side to compare.
DELL_FULL_SWEEP = {
    ("all_reduce",     2): 61.28, ("all_reduce",     3): 75.02, ("all_reduce",     4): 166.48,
    ("all_reduce",     5): 38.36, ("all_reduce",     6): 38.42, ("all_reduce",     7): 38.21,
    ("all_reduce",     8): 381.27,
    ("all_gather",     2): 60.58, ("all_gather",     3): 71.09, ("all_gather",     4): 158.72,
    ("all_gather",     5): 35.41, ("all_gather",     6): 34.90, ("all_gather",     7): 34.85,
    ("all_gather",     8): 365.75,
    ("reduce_scatter", 2): 60.65, ("reduce_scatter", 3): 71.04, ("reduce_scatter", 4): 165.06,
    ("reduce_scatter", 5): 39.57, ("reduce_scatter", 6): 39.65, ("reduce_scatter", 7): 40.47,
    ("reduce_scatter", 8): 407.69,
    ("broadcast",      2): 63.52, ("broadcast",      3): 68.09, ("broadcast",      4): 169.19,
    ("broadcast",      5): 34.10, ("broadcast",      6): 33.92, ("broadcast",      7): 33.83,
    ("broadcast",      8): 377.27,
    ("reduce",         2): 72.87, ("reduce",         3): 86.46, ("reduce",         4): 197.44,
    ("reduce",         5): 43.60, ("reduce",         6): 42.93, ("reduce",         7): 43.08,
    ("reduce",         8): 358.49,
    ("gather",         2): 72.07, ("gather",         3): 78.27, ("gather",         4): 211.64,
    ("gather",         5): 69.38, ("gather",         6): 68.83, ("gather",         7): 70.29,
    ("gather",         8): 444.15,
    ("scatter",        2): 63.11, ("scatter",        3): 71.48, ("scatter",        4): 191.63,
    ("scatter",        5): 65.35, ("scatter",        6): 65.61, ("scatter",        7): 66.36,
    ("scatter",        8): 426.40,
    ("alltoall",       2): 58.40, ("alltoall",       3): 61.79, ("alltoall",       4): 155.21,
    ("alltoall",       5): 44.14, ("alltoall",       6): 45.62, ("alltoall",       7): 44.26,
    ("alltoall",       8): 360.90,
    ("sendrecv",       2): 59.21, ("sendrecv",       3): 60.32, ("sendrecv",       4): 60.59,
    ("sendrecv",       5): 43.83, ("sendrecv",       6): 43.77, ("sendrecv",       7): 43.44,
    ("sendrecv",       8): 53.24,
}
# Our sweep caps alltoall/alltoallv at ALLTOALL_MAX (default 4 GiB, see run-rccl-all.sh)
# instead of 8 GiB, to survive the N=5 OOM that killed Dell Cloud's alltoallv run. Flag it
# in the comparison rather than silently comparing different message sizes.
SMALLER_TOP_SIZE = {"alltoall", "alltoallv"}

# NVIDIA reference fabrics -- PUBLISHED SPEC ONLY, not measured. No NCCL run exists on
# either machine in this repo, so the measured column is deliberately empty: quoting a
# third-party busbw number next to ours would not be a like-for-like comparison.
NV_FABRICS = [
    ("NVIDIA H100 SXM (ref)", "NVLink 4 + NVSwitch", "switched all-to-all",
     "25 GB/s x 18 links", 900.0, 450.0),
    ("NVIDIA B200 SXM (ref)", "NVLink 5 + NVSwitch", "switched all-to-all",
     "50 GB/s x 18 links", 1800.0, 900.0),
]


def ceiling_gbps(coll: str, n: int):
    """Max bandwidth this collective can physically engage at this arity, per direction.

    Not the GPU's full 7-link aggregate: sendrecv only lights one link per pair, while a
    ring at N drives min(N, 7) links concurrently. Approximation -- it reproduces the two
    anchor points in dell-cloud's table (N=8 -> 537.6, N=4 -> ~307).
    """
    if coll == "sendrecv":
        return LINK_UNIDIR, "1 link x 1 direction"
    links = min(n, MAX_LINKS)
    if coll in RING_COLLS:
        return LINK_UNIDIR * links, f"{links}-link ring x 1 direction"
    if coll in PAIRWISE_COLLS:
        return LINK_UNIDIR * links, f"{links} concurrent pairwise links"
    return None, ""


def parse(path: Path):
    """Return {size_bytes: busbw} from the in-place columns of a rccl-tests table."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < 8 or not f[0].isdigit():
            continue
        try:
            out[int(f[0])] = float(f[-2])   # in-place busbw is second-to-last column
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in a.dirs:
        for log in sorted(d.glob("*.log")):
            m = NAME.match(log.name)
            if not m:
                continue
            pts = parse(log)
            if not pts:
                continue
            top = max(pts)
            rows.append({"run": d.name, "collective": m["coll"], "config": m["cfg"] or "default",
                         "gpus": int(m["n"]), "max_size_bytes": top,
                         "busbw_at_max_GBps": pts[top],
                         "peak_busbw_GBps": max(pts.values()),
                         "n_sizes": len(pts)})
    if not rows:
        raise SystemExit("no rccl logs parsed")

    with (a.out / "rccl.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def table(sel, key, title):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return []
        ks = sorted({r[key] for r in sub})
        ns = sorted({r["gpus"] for r in sub})
        idx = {(r[key], r["gpus"]): r["busbw_at_max_GBps"] for r in sub}
        L = [f"### {title}", "",
             f"| {key} | " + " | ".join(f"N={n}" for n in ns) + " | cliff |",
             "|---|" + "---:|" * (len(ns) + 1)]
        for k in ks:
            vals = [idx.get((k, n)) for n in ns]
            cells = [f"{v:.1f}" if v else "-" for v in vals]
            p2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) == 0]
            np2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) != 0]
            flag = "-"
            if p2 and np2:
                ratio = min(np2) / (sum(p2) / len(p2))
                flag = f"**{(1 - ratio) * 100:.0f}% down**" if ratio < 0.85 else "none"
            L.append(f"| {k} | " + " | ".join(cells) + f" | {flag} |")
        return L + [""]

    L = ["# RCCL Collective Communications — MI355X x8, XGMI", "",
         "System: 8 x AMD Instinct MI355X (gfx950), ROCm 7.14, XGMI all-to-all (K8 mesh, "
         "every pair 1 hop). Built natively for gfx950 with no `HSA_OVERRIDE_GFX_VERSION`, "
         "so absolute numbers may exceed the Dell Cloud gfx942-override run.", "",
         f"Source runs: {', '.join(sorted({r['run'] for r in rows}))}", "",
         "`busbw` is steady-state bytes crossing the wire per unit time, normalized for each "
         "algorithm's theoretical data movement -- the comparable metric across N and across "
         "collectives. All figures below are busbw at the top message size.", "",
         "## 1. Measured results", ""]

    L += table(lambda r: r["config"] == "default", "collective", "1.1 Full collective sweep")

    # ---- Dell Cloud vs AMD Cloud, full sweep, one table --------------------------
    dflt_all = {(r["collective"], r["gpus"]): r["busbw_at_max_GBps"]
                for r in rows if r["config"] == "default"}
    common = sorted({c for c, n in DELL_FULL_SWEEP} & {c for c, n in dflt_all})
    if common:
        ns_full = sorted({n for c, n in DELL_FULL_SWEEP if c in common})
        L += ["### 1.1a Dell Cloud vs AMD Cloud — full sweep (busbw GB/s at top message size)",
              "",
              "Same silicon, same fabric (8 x MI355X, XGMI 4th gen K8 mesh) on both hosts; "
              "the only difference is software (ROCm 7.2.3 + gfx942 alias on Dell Cloud vs "
              "ROCm 7.14 native gfx950 here). Each cell is `Dell / AMD (AMD÷Dell)`.",
              "",
              "| Collective | " + " | ".join(f"N={n}" for n in ns_full) + " |",
              "|---|" + "---|" * len(ns_full)]
        for c in common:
            note = "‡" if c in SMALLER_TOP_SIZE else ""
            cells = []
            for n in ns_full:
                d, m = DELL_FULL_SWEEP.get((c, n)), dflt_all.get((c, n))
                if d is None or m is None:
                    cells.append("-")
                    continue
                ratio = m / d
                flag = "**" if ratio < 0.85 or ratio > 1.5 else ""
                cells.append(f"{d:.1f}/{flag}{m:.1f}{flag} ({ratio:.2f}x)")
            L.append(f"| {c}{note} | " + " | ".join(cells) + " |")
        if any(c in SMALLER_TOP_SIZE for c in common):
            L += ["", "‡ measured here at a smaller top message size than Dell Cloud's 8 GiB "
                  "(both sides cap `alltoall`/`alltoallv` early to survive the N=5 OOM that "
                  "killed Dell Cloud's alltoallv run — see run-rccl-all.sh `ALLTOALL_MAX`). "
                  "busbw plateaus well before 8 GiB for every collective measured (Dell "
                  "Cloud's own finding, summary-rccl.md §1.1), so the smaller cap should "
                  "still land in the flat region, but it is not a strictly identical "
                  "measurement and is flagged rather than presented as one."]
        gap = [(c, n, dflt_all[(c, n)] / DELL_FULL_SWEEP[(c, n)])
               for c in common for n in ns_full
               if (c, n) in dflt_all and (c, n) in DELL_FULL_SWEEP]
        if gap:
            best = max(gap, key=lambda t: t[2])
            worst = min(gap, key=lambda t: t[2])
            L += ["", f"Ratio ranges from **{worst[2]:.2f}x** (`{worst[0]}` N={worst[1]}) to "
                  f"**{best[2]:.2f}x** (`{best[0]}` N={best[1]}). Bold cells are >1.5x or "
                  f"<0.85x — outside what run-to-run noise on identical hardware would "
                  f"explain.", ""]

    # ---- Infinity Fabric spec vs measured ---------------------------------------
    L += ["### 1.2 Infinity Fabric paper spec vs measured ceilings", "",
          "Each GPU has 7 xGMI links wired point-to-point to the other 7 GPUs. On-node "
          "bandwidth telemetry reports N/A on this driver build, so these are AMD's published "
          "MI350-series peaks:", "",
          "| Quantity | Spec |", "|---|---:|",
          f"| Per xGMI link, bidirectional | **{LINK_BIDIR} GB/s** |",
          f"| Per xGMI link, per direction | {LINK_UNIDIR} GB/s |",
          f"| Per-GPU aggregate (x7 links), bidirectional | **{LINK_BIDIR * MAX_LINKS:.1f} GB/s** |",
          f"| Per-GPU aggregate (x7 links), per direction | {LINK_UNIDIR * MAX_LINKS:.1f} GB/s |", "",
          "The comparable ceiling depends on how many links the *specific* collective engages: "
          "sendrecv lights one link per pair, while a ring at N drives min(N, 7) links "
          "concurrently. Comparing every row to the full 7-link aggregate would understate "
          "small-N results.", "",
          "| Collective | N | Measured (GB/s) | Ceiling (GB/s) | Basis | Achieved |",
          "|---|---:|---:|---:|---|---:|"]
    for r in sorted((r for r in rows if r["config"] == "default"),
                    key=lambda r: (r["collective"], r["gpus"])):
        c, basis = ceiling_gbps(r["collective"], r["gpus"])
        if not c:
            continue
        pct = 100 * r["busbw_at_max_GBps"] / c
        mark = "**" if pct < 25 else ""
        L.append(f"| {r['collective']} | {r['gpus']} | {r['busbw_at_max_GBps']:.2f} | "
                 f"{c:.1f} | {basis} | {mark}{pct:.0f}%{mark} |")
    L += ["", "Rows below 25% of their ceiling are bolded: at that level the arity is not "
          "constructing a usable communication pattern, rather than merely running "
          "inefficiently.", ""]

    # ---- cross-machine interconnect comparison ----------------------------------
    dflt = {(r["collective"], r["gpus"]): r["busbw_at_max_GBps"]
            for r in rows if r["config"] == "default"}
    xgmi_bidir = LINK_BIDIR * MAX_LINKS
    xgmi_unidir = LINK_UNIDIR * MAX_LINKS
    ar8 = dflt.get(("all_reduce", 8))

    L += ["### 1.3 Interconnect comparison — Dell Cloud vs AMD Cloud vs NVIDIA reference", "",
          "Dell Cloud and AMD Cloud are the **same fabric on the same silicon** (8 x MI355X, "
          "XGMI 4th gen, K8 direct mesh); only the software stack differs. The NVIDIA rows are "
          "**published spec only** — no NCCL run exists on either machine in this repo, so "
          "quoting someone else's busbw beside ours would not be like-for-like.", "",
          "| Machine | Fabric | Topology | Per-link (bidir) | Per-GPU aggregate (bidir) | "
          "Per-GPU (per direction) | Measured AllReduce N=8 | % of ceiling |",
          "|---|---|---|---|---:|---:|---:|---:|"]
    L.append(f"| Dell Cloud — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | "
             f"{LINK_BIDIR} GB/s x7 | {xgmi_bidir:.1f} GB/s | {xgmi_unidir:.1f} GB/s | "
             f"{DELL_RCCL[('all_reduce', 8)]:.2f} GB/s | "
             f"{100 * DELL_RCCL[('all_reduce', 8)] / xgmi_unidir:.0f}% |")
    L.append(f"| **AMD Cloud (this host)** — 8x MI355X | Infinity Fabric (XGMI) 4th gen | "
             f"direct mesh (K8, 1 hop) | {LINK_BIDIR} GB/s x7 | {xgmi_bidir:.1f} GB/s | "
             f"{xgmi_unidir:.1f} GB/s | "
             f"{f'**{ar8:.2f} GB/s**' if ar8 else '_not yet measured_'} | "
             f"{f'{100 * ar8 / xgmi_unidir:.0f}%' if ar8 else '—'} |")
    for name, fabric, topo, per_link, bidir, unidir in NV_FABRICS:
        L.append(f"| {name} — 8x GPU | {fabric} | {topo} | {per_link} | {bidir:.1f} GB/s | "
                 f"{unidir:.1f} GB/s | _not measured (spec only)_ | — |")

    L += ["", "Reading:", "",
          f"- **B200's NVLink 5 has ~1.67x the per-GPU fabric bandwidth of MI355X's XGMI** "
          f"({NV_FABRICS[1][4]:.0f} vs {xgmi_bidir:.0f} GB/s bidirectional). H100's NVLink 4 is "
          f"slightly *below* MI355X ({NV_FABRICS[0][4]:.0f} GB/s) — the dell-cloud readme's "
          f"\"comparable to NVLink 4\" characterisation is right for H100 and wrong for B200.",
          "- **The architectural difference matters more than the headline number.** NVIDIA "
          "routes through an NVSwitch, so any subset of GPUs gets full switched all-to-all "
          "bandwidth. AMD's mesh is direct point-to-point, which is why ring construction — and "
          "therefore the collective arity N — determines how much of the fabric is reachable.",
          f"- That is the root of the non-power-of-2 cliff: dell-cloud measured {DELL_CLIFF_NOTE} "
          f"for AllReduce, against {DELL_RCCL[('all_reduce', 8)]:.0f} GB/s at N=8. A switched "
          f"fabric has no equivalent failure mode, which is why NVIDIA stopped seeing these "
          f"cliffs after DGX-1/P100 and why AMD's structural fix is UALink in MI400 rather than "
          f"more MSCCL plans.", ""]

    if dflt:
        L += ["### 1.4 Same-silicon comparison vs Dell Cloud", "",
              "Both hosts are 8 x MI355X. Dell Cloud ran ROCm 7.2.3 with the gfx942 alias; this "
              "host runs ROCm 7.14 with native gfx950 code objects.", "",
              "| Collective | N | Dell Cloud | AMD Cloud | AMD/Dell |", "|---|---:|---:|---:|---:|"]
        for (coll, n), dv in sorted(DELL_RCCL.items()):
            mv = dflt.get((coll, n))
            L.append(f"| {coll} | {n} | {dv:.2f} | "
                     f"{f'{mv:.2f}' if mv else '-'} | "
                     f"{f'**{mv / dv:.2f}x**' if mv else '-'} |")
        L += [""]

    # ---- config comparison -------------------------------------------------------
    cfgs = {r["config"] for r in rows if r["config"] != "default"}
    if cfgs:
        L += ["## 2. Config sweep — which knob recovers a cliff", ""]
        for coll in sorted({r["collective"] for r in rows if r["config"] != "default"}):
            L += table(lambda r, c=coll: r["collective"] == c, "config",
                       f"`{coll}` — config comparison")

    L += ["## 3. How to read the cliff column", "",
          "`X% down` means the worst non-power-of-2 N is that much below the mean of the "
          "power-of-2 Ns.", "",
          "- If the `default` config cliffs but `tree`/`ring`/`no_mscll` do not, the recovering "
          "knob is a one-env-var workaround and should be adopted.",
          "- If nothing recovers it, the gap is missing RCCL tuning for those arities on "
          "gfx950 — RCCL has failed to construct a valid ring, and the fix is upstream.",
          "- Either way the attribution to the *algorithm layer* rests on Part A's RVS `pbqt` "
          "(peer-to-peer XGMI) and `pebb` (PCIe) runs coming back clean. Without that, a cliff "
          "could equally be a bad link.", "",
          "This matters for training: a Ring AllReduce at N=8 is the realistic upper bound for "
          "data-parallel gradient sync on this box — no Megatron dist-opt tuning can exceed it.",
          "", "## 4. Reference", "", "### 4.1 RCCL algorithm selection", "",
          "| Collective | Ring | Tree | PAT | MSCCL | Pairwise sendrecv |",
          "|---|---|---|---|---|---|",
          "| AllReduce | Default large | Default small (degraded on mesh) | — | If plan exists | — |",
          "| AllGather | Default | Falls back to Ring | Available | If plan exists | — |",
          "| ReduceScatter | Default | Falls back to Ring | Available | If plan exists | — |",
          "| Broadcast | Default large | Default small | — | — | — |",
          "| Reduce | Default large | Default small | — | — | — |",
          "| Gather / Scatter | — | — | — | — | Default |",
          "| AllToAll | — | — | — | If plan exists | Default |",
          "| SendRecv | — | — | — | — | Direct |", "",
          "- **NVLS** (in-network reduction) is not applicable on AMD until UALink ships. "
          "**PAT** is a switched-fabric path, not active on an xGMI mesh.",
          "- Forcing `NCCL_ALGO=Tree` on AllGather / ReduceScatter is silently equivalent to Ring.",
          "- MSCCL plans ship in `/opt/rocm/share/rccl/msccl-algorithms/`; everything without a "
          "plan falls through to Ring or pairwise sendrecv.", "",
          "### 4.2 Configuration knobs", "",
          "| Variable | Value used here | Effect |", "|---|---|---|",
          "| `NCCL_ALGO` | `Ring,Tree` | Algorithm pool. `Tree` only affects AllReduce / Broadcast / Reduce. |",
          "| `NCCL_PROTO` | `Simple,LL,LL128` | Wire protocol; `Simple` is effectively the default at large message size. |",
          "| `RCCL_MSCCL_ENABLE` | `1` | Toggle MSCCL plan dispatch. |",
          "| `NCCL_P2P_DISABLE` | `0` | `1` forces host-SHM staging; debug only. |",
          "| `NCCL_SHM_DISABLE` | `0` | Leave on. |",
          "| `NCCL_IB_DISABLE` | `1` | Single-node run, no IB. |",
          "| `NCCL_SOCKET_IFNAME` | `lo` | Bootstrap over loopback. |",
          "| `NCCL_DEBUG` | `WARN` | `INFO` prints algorithm + channel count per call. |",
          "| `HSA_OVERRIDE_GFX_VERSION` | **unset** | Deliberately native gfx950; the gfx942 alias would undercount. |",
          "", "### 4.3 Process model caveat", "",
          "rccl-tests runs **one process driving N GPUs** (`-g N`, `MPI=0`), while real training "
          "and Primus' `benchmark rccl` run **N processes with 1 GPU each**. These take different "
          "code paths inside RCCL. If Part C's collective numbers disagree with these, the "
          "process model is the first suspect.", "",
          "## 5. Source data", "", "| What | Where |", "|---|---|",
          "| Raw rccl-tests stdout | `logs/rccl/rccl_*/<coll>_n<N>.log` |",
          "| Config sweep logs | `logs/rccl/rccl_tests_*/<coll>_<cfg>_n<N>.log` |",
          "| Per-run summary | `logs/rccl/rccl_*/rccl_summary.txt` |",
          "| This table as CSV | `results/rccl.csv` |",
          "| Figure | `results/rccl_busbw.png` |", ""]

    (a.out / "rccl.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rccl.md and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
