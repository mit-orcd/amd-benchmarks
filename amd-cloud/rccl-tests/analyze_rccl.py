#!/usr/bin/env python3
"""Parse rccl-tests logs into results/rccl.{md,csv} and flag the non-power-of-2 cliff.

Usage: analyze_rccl.py <log_dir> [<log_dir> ...] -o results

Parses the raw logs directly (not the summary text), so it works on partial runs.
Handles both naming schemes: <coll>_n<N>.log and <coll>_<config>_n<N>.log
"""
import argparse, csv, re
from pathlib import Path

NAME = re.compile(
    r"^(?P<coll>[a-z_]+?)(?:_(?P<cfg>default|tree|ring|no_mscll|proto_simple))?_n(?P<n>\d+)\.log$")


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
        L = [f"## {title}", "",
             f"| {key} | " + " | ".join(f"N={n}" for n in ns) + " | cliff |",
             "|" + "---|" * (len(ns) + 2)]
        for k in ks:
            vals = [idx.get((k, n)) for n in ns]
            cells = [f"{v:.1f}" if v else "-" for v in vals]
            # cliff = worst non-power-of-2 N vs the mean of the power-of-2 Ns
            p2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) == 0]
            np2 = [v for n, v in zip(ns, vals) if v and (n & (n - 1)) != 0]
            flag = "-"
            if p2 and np2:
                ratio = min(np2) / (sum(p2) / len(p2))
                flag = f"{(1 - ratio) * 100:.0f}% down" if ratio < 0.85 else "none"
            L.append(f"| {k} | " + " | ".join(cells) + f" | {flag} |")
        return L + [""]

    L = ["# RCCL collective bandwidth — MI355X x8, XGMI (busbw at top message size)", "",
         "Built natively for gfx950 against host ROCm 7.14 (no `HSA_OVERRIDE_GFX_VERSION`), "
         "so absolute numbers may exceed the Dell Cloud gfx942-override run.", ""]
    L += table(lambda r: r["config"] == "default", "collective", "All collectives (default config)")
    for coll in sorted({r["collective"] for r in rows if r["config"] != "default"}):
        L += table(lambda r, c=coll: r["collective"] == c, "config", f"`{coll}` — config comparison")
    L += ["## How to read the cliff column", "",
          "`X% down` means the worst non-power-of-2 N is that much below the mean of the "
          "power-of-2 Ns. If the `default` config cliffs but `tree`/`ring`/`no_mscll` do not, "
          "the recovering knob is a one-env-var workaround. If nothing recovers it, the gap is "
          "missing RCCL tuning for those arities on gfx950 — and per `rccl-tests.md`, a clean "
          "RVS `pbqt` run (Part A) is what lets you attribute it to the algorithm layer rather "
          "than to the fabric.", ""]
    (a.out / "rccl.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}/rccl.md and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
