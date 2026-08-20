#!/usr/bin/env python3
"""Generate results/kimi-k3-single-stream.md from the low-concurrency kernel-path sweep.

Usage: analyze_single_stream.py <kimi_single_stream_dir> -o <results_dir>

Reports PER-REQUEST speed (1000/TPOT) as the primary metric, with aggregate throughput
secondary -- the inverse of every other analyzer here, because this experiment exists to
test whether one user's decode rate can be moved at all.

K1_mad_default is the control: arms are compared to K1, NOT to Run A's 46.6 tok/s, since
Run A used a different image (rocm/atom-dev:latest) and that comparison would confound
image with kernel path. Run A is shown for context and labelled as such.
"""
import argparse, csv, json, re, sys
from pathlib import Path

CONTROL = "K1_mad_default"
RUN_A_C1 = 46.6      # tok/s per request at c=1, Run A, rocm/atom-dev:latest
RUN_A_TPOT_C1 = 21.48

ARM_DESC = {
    "K1_mad_default":  "MAD baseline (control)",
    "K2_triton_moe":   "`ATOM_USE_TRITON_MOE=1` — MoE kernel path",
    "K3_aiter_attn":   "`ATOM_USE_UNIFIED_ATTN=0`, `ATOM_FORCE_ATTN_TRITON=0` — attention path",
    "K4_grouped_gemm": "`ATOM_USE_TRITON_GEMM=0`, `AITER_USE_GROUPED_GEMM=1` — GEMM path",
}


def load_arm(d: Path):
    rows = {}
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        c = int(data.get("max_concurrency") or re.sub(r"\D", "", j.stem) or 0)
        tpot = data.get("median_tpot_ms")
        rows[c] = dict(
            tpot=tpot,
            per_req=(1000.0 / tpot) if tpot else None,
            agg=data.get("output_throughput"),
            ttft=data.get("median_ttft_ms"),
            p99_tpot=data.get("p99_tpot_ms"),
        )
    return rows


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    arms = {}
    for d in sorted(p for p in a.sweep.iterdir() if p.is_dir()):
        r = load_arm(d)
        if r:
            arms[d.name] = r
    if not arms:
        sys.exit(f"no usable result JSON under {a.sweep}")

    concs = sorted({c for r in arms.values() for c in r})
    names = sorted(arms, key=lambda n: (n != CONTROL, n))

    with (a.out / "kimi-k3-single-stream.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "concurrency", "median_tpot_ms", "per_request_tok_s",
                    "aggregate_tok_s", "median_ttft_ms", "p99_tpot_ms"])
        for n in names:
            for c in sorted(arms[n]):
                v = arms[n][c]
                w.writerow([n, c, v["tpot"], v["per_req"], v["agg"], v["ttft"], v["p99_tpot"]])

    L = []
    A = L.append
    A("# Kimi-K3 — single-stream (per-request) speed vs kernel path")
    A("")
    A(f"Source run: `{a.sweep.name}`. Low-concurrency sweep across kernel-path configurations,")
    A("all on the MAD-pinned image, TP=8, ISL/OSL 1024/1024.")
    A("")
    A("**Primary metric is per-request tok/s (`1000 / median TPOT`)** — the decode rate one user")
    A("experiences — not aggregate throughput. This is the only experiment in the Kimi-K3 set")
    A("that targets it; every other one measured aggregate tok/s, where batching dominates.")
    A("")
    A("## Arms")
    A("")
    A("| Arm | Change from MAD baseline |")
    A("|---|---|")
    for n in names:
        A(f"| `{n}` | {ARM_DESC.get(n, 'see logs/env.txt')} |")
    A("")
    A("Each arm flips exactly one kernel-path decision, so a difference is attributable.")
    A("")

    A("## Per-request tok/s (higher is better)")
    A("")
    A("| Concurrency | " + " | ".join(f"`{n}`" for n in names) + " |")
    A("|---:|" + "---:|" * len(names))
    for c in concs:
        cells = [f(arms[n].get(c, {}).get("per_req")) for n in names]
        A(f"| {c} | " + " | ".join(cells) + " |")
    A("")

    A("## Median TPOT (ms, lower is better)")
    A("")
    A("| Concurrency | " + " | ".join(f"`{n}`" for n in names) + " |")
    A("|---:|" + "---:|" * len(names))
    for c in concs:
        cells = [f(arms[n].get(c, {}).get("tpot"), 2) for n in names]
        A(f"| {c} | " + " | ".join(cells) + " |")
    A("")

    A("## Aggregate tok/s (secondary)")
    A("")
    A("| Concurrency | " + " | ".join(f"`{n}`" for n in names) + " |")
    A("|---:|" + "---:|" * len(names))
    for c in concs:
        cells = [f(arms[n].get(c, {}).get("agg")) for n in names]
        A(f"| {c} | " + " | ".join(cells) + " |")
    A("")

    # ---- verdict, measured against the control ----
    A("## Reading")
    A("")
    ctrl = arms.get(CONTROL)
    c1 = concs[0]
    if ctrl and c1 in ctrl and ctrl[c1]["per_req"]:
        base = ctrl[c1]["per_req"]
        A(f"Control `{CONTROL}` at c={c1}: **{f(base)} tok/s per request** "
          f"(TPOT {f(ctrl[c1]['tpot'],2)} ms).")
        A("")
        deltas = []
        for n in names:
            if n == CONTROL:
                continue
            v = arms[n].get(c1, {}).get("per_req")
            if v:
                deltas.append((n, v, 100.0 * (v / base - 1)))
        if deltas:
            A(f"| Arm | per-request tok/s @ c={c1} | vs control |")
            A("|---|---:|---:|")
            for n, v, pct in deltas:
                A(f"| `{n}` | {f(v)} | {pct:+.1f}% |")
            A("")
            best_n, best_v, best_pct = max(deltas, key=lambda x: x[1])
            if best_pct >= 5:
                A(f"**A kernel path moves single-stream speed: `{best_n}` is {best_pct:+.1f}% "
                  f"over the control.** Configuration-level tuning is therefore not exhausted "
                  f"for this metric. Worth confirming with a repeat before acting on it — a "
                  f"single measurement at one concurrency is thin evidence for a change of "
                  f"this size.")
            elif best_pct <= -5:
                A(f"**Every arm is at or below the control** (best: `{best_n}` at "
                  f"{best_pct:+.1f}%). The MAD kernel set is already the better choice at low "
                  f"batch, and these knobs do not offer a single-stream win.")
            else:
                A(f"**No kernel path meaningfully changes single-stream speed** — the full "
                  f"spread is within ±5% of the control. This is a clean negative result and "
                  f"it is informative: it localizes the per-request cost to things no "
                  f"environment variable can reach — kernel launch overhead, the 186 "
                  f"serialized all-reduces (fixed by TP=8 × 93 layers), and the sequential "
                  f"dependency in the 69 KDA layers. **Configuration-level tuning for "
                  f"per-request speed is closed**; further gains require ATOM/AITER changes.")
            A("")

        A(f"**Context — Run A measured {RUN_A_C1} tok/s at c=1** (TPOT {RUN_A_TPOT_C1} ms) on "
          f"`rocm/atom-dev:latest`. That is a *different image*, so it is not a clean "
          f"comparison against these arms; `{CONTROL}` is the matched control. The two are "
          f"listed together only to show the regime is consistent.")
        A("")

    A("## The headroom this was testing against")
    A("")
    A("At c=1 a decode step reads ~3.4 GB of weights per GPU. At the ~2.2 TB/s effective rate")
    A("implied by the c=64 measurement (116 GB / 51.7 ms), that is **~1.5 ms** of weight")
    A("reading against a measured TPOT of ~21 ms — so **~93% of a single-request step is not")
    A("weight traffic**. That 93% is serialization, and this experiment asks how much of it is")
    A("reachable from configuration. Whatever the answer, the ~1.5 ms figure is a floor and")
    A("not a target: kernel dispatch, the KDA dependency chain, and a real minimum collective")
    A("latency are all irreducible.")
    A("")
    A("Levers deliberately **not** tested here, because they are unavailable rather than")
    A("untried: reducing the 186 all-reduces (fixed by TP=8 × 93 layers, no flag); TP<8 with")
    A("replicas (1.5 TB of weights against 2.3 TB of HBM forces TP=8 on one node);")
    A("speculative decoding / MTP (`num_nextn_predict_layers = 0` — no MTP heads shipped);")
    A("HIP graphs (already enabled, server log reports `cudagraph=True`).")
    A("")
    A("## Source data")
    A("")
    A("| What | Where |")
    A("|---|---|")
    A(f"| Per-arm JSON / logs | `{a.sweep.name}/<arm>/c<N>.{{json,log}}` |")
    A(f"| Per-arm env (exact vars) | `{a.sweep.name}/<arm>/env.txt` |")
    A(f"| Server logs | `{a.sweep.name}/<arm>/atom_server.log` |")
    A(f"| Driver state | `{a.sweep.name}/STATE.txt` |")
    A("| This data as CSV | `results/kimi-k3-single-stream.csv` |")
    A("| Context and rationale | `kimi-k3-improve.md` §4 *Improving per-request speed* |")
    A("")

    (a.out / "kimi-k3-single-stream.md").write_text("\n".join(L))
    print(f"wrote {a.out/'kimi-k3-single-stream.md'} ({len(arms)} arms, {len(concs)} concurrencies)")


if __name__ == "__main__":
    main()
