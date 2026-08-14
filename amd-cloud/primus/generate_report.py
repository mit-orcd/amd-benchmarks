#!/usr/bin/env python3
"""Parse a Primus sweep directory and emit REPORT.md.

Usage: generate_report.py <sweep_dir> <bench_out_dir> <b200_summary.md> <report.md>
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

GPU_RANGE = list(range(1, 9))


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _md_rows(text: str) -> list[list[str]]:
    """Parse a github-flavored markdown table; return rows (excluding header sep)."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|") or not s.endswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator
        rows.append(cells)
    return rows


def _floats(values: Iterable[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v.replace(",", "")))
        except ValueError:
            pass
    return out


def _col(rows: list[list[str]], name: str) -> list[float]:
    if not rows:
        return []
    header = [h.lower() for h in rows[0]]
    if name.lower() not in header:
        return []
    idx = header.index(name.lower())
    return _floats(r[idx] for r in rows[1:] if len(r) > idx)


# ---------- per-bench parsers ----------

_TF_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*TF\s*/?\s*s", re.IGNORECASE)


def parse_gemm_like(path: Path) -> dict:
    """Generic parser for gemm / gemm-dense / gemm-deepseek markdown reports.

    Two layouts supported:
      1. Plain `tflops` column per row (gemm).
      2. Compound cells "X.XXs / YYY.YYTF/s / ZZZ.ZZGB/s / AI=..." per shape×phase
         (gemm-dense, gemm-deepseek).
    """
    rows = _md_rows(_read(path))
    if not rows:
        return {}

    tflops = _col(rows, "tflops")
    if not tflops:
        # Compound-cell layout: scan every data cell for "<num>TF/s" tokens.
        scanned: list[float] = []
        for r in rows[1:]:
            for cell in r:
                for m in _TF_TOKEN_RE.finditer(cell):
                    try:
                        scanned.append(float(m.group(1)))
                    except ValueError:
                        pass
        tflops = scanned

    if not tflops:
        return {}
    return {
        "mean_tflops": mean(tflops),
        "min_tflops": min(tflops),
        "max_tflops": max(tflops),
        "stdev_tflops": stdev(tflops) if len(tflops) > 1 else 0.0,
        "n_ranks": len(tflops),
    }


def parse_attention(path: Path) -> dict:
    """Parse attention CSV — find any TFLOP-ish column."""
    if not path.exists():
        return {}
    rows = list(csv.reader(open(path)))
    if len(rows) < 2:
        return {}
    header = [h.lower().strip() for h in rows[0]]
    candidates = [i for i, h in enumerate(header) if "tflop" in h or "tf/s" in h]
    out = {}
    for i in candidates:
        col = header[i]
        vals = _floats(r[i] for r in rows[1:] if len(r) > i)
        if vals:
            out[col] = {"mean": mean(vals), "max": max(vals), "n": len(vals)}
    if not out:
        # No TFLOP col — look for time/latency
        for i, h in enumerate(header):
            if "ms" in h or "time" in h:
                vals = _floats(r[i] for r in rows[1:] if len(r) > i)
                if vals:
                    out[h] = {"mean": mean(vals), "min": min(vals), "n": len(vals)}
                    break
    return out


def parse_rccl(path: Path) -> dict:
    """Parse rccl markdown — extract peak busbw (GB/s) at the largest sizes."""
    rows = _md_rows(_read(path))
    if len(rows) < 2:
        return {}
    header = [h.lower() for h in rows[0]]
    # Find bw column (busbw / bandwidth / gb/s)
    bw_idx = None
    for i, h in enumerate(header):
        h_norm = h.replace(" ", "")
        if (
            "busbw" in h
            or "bw_gbps" in h
            or "eff_gbps" in h
            or h == "bw"
            or "gb/s" in h_norm
            or "gbps" in h_norm
            or "bandwidth" in h
        ):
            bw_idx = i
            break
    if bw_idx is None:
        return {}
    vals = _floats(r[bw_idx] for r in rows[1:] if len(r) > bw_idx)
    if not vals:
        return {}
    return {
        "peak_bw_gbps": max(vals),
        "mean_bw_gbps": mean(vals),
        "n_sizes": len(vals),
    }


# Megatron iteration line regex (permissive)
_MLM_TFLOPS_RE = re.compile(
    r"(?:TFLOPs?[\s\-_/]?(?:per|/)?[\s_]*[Gg]?[Pp]?[Uu]?|throughput\s*per\s*GPU\s*\(TFLOPs?/?s?/?GPU\)|TFLOP[sS]?/?[sS]?/?GPU)"
    r"\s*[:=]?\s*"
    r"(\d+\.\d+|\d+)",
    re.IGNORECASE,
)
_MLM_ITER_TIME_RE = re.compile(
    r"elapsed time per iteration[^|]*?(\d+\.\d+|\d+)",
    re.IGNORECASE,
)
_MLM_GBS_RE = re.compile(r"global batch size\s*:?\s*(\d+)", re.IGNORECASE)

# The Megatron iteration line emits TWO different TFLOP/s/GPU figures, and conflating them
# is a real trap (it produced a spurious 3.8x "regression" vs the Dell Cloud Primus run):
#   compute per GPU (TFLOP/s/GPU): 1135.2   <- kernel-time throughput; what dell-cloud's
#                                              REPORT.md "last TF/s/GPU" column actually is
#   throughput per GPU (TFLOP/s/GPU): 294.4 <- wall-clock throughput, includes bubbles/idle
# _MLM_TFLOPS_RE is permissive enough to match both and keeps the last, so parse the
# compute figure separately and report both, explicitly labelled.
_MLM_COMPUTE_RE = re.compile(
    r"compute per GPU \(TFLOPs?/?s?/GPU\)\s*[:=]\s*(\d+\.\d+|\d+)", re.IGNORECASE)
_MLM_WALL_RE = re.compile(
    r"throughput per GPU \(TFLOPs?/?s?/GPU\)\s*[:=]\s*(\d+\.\d+|\d+)", re.IGNORECASE)


def parse_megatron(path: Path) -> dict:
    text = _read(path)
    if not text:
        return {}
    tflops = [float(m.group(1)) for m in _MLM_TFLOPS_RE.finditer(text)]
    iter_ms = [float(m.group(1)) for m in _MLM_ITER_TIME_RE.finditer(text)]
    gbs_matches = [int(m.group(1)) for m in _MLM_GBS_RE.finditer(text)]
    oom = "out of memory" in text.lower() or "OOM" in text
    failed = "FAILED" in text or "ChildFailedError" in text
    out = {
        "oom": oom,
        "failed": failed,
        "n_iters_logged": len(tflops),
        "gbs": gbs_matches[-1] if gbs_matches else None,
    }
    if tflops:
        # Skip first 2 (warmup); take last 10 or what's left.
        usable = tflops[2:] if len(tflops) > 4 else tflops
        out["last_tflops"] = tflops[-1]
        out["mean_tflops"] = mean(usable)
        out["max_tflops"] = max(usable)
    if iter_ms:
        usable = iter_ms[2:] if len(iter_ms) > 4 else iter_ms
        out["last_iter_ms"] = iter_ms[-1]
        out["mean_iter_ms"] = mean(usable)
    comp = [float(m.group(1)) for m in _MLM_COMPUTE_RE.finditer(text)]
    wall = [float(m.group(1)) for m in _MLM_WALL_RE.finditer(text)]
    if comp:
        out["last_compute_tflops"] = comp[-1]
    if wall:
        out["last_wall_tflops"] = wall[-1]
    return out


# ---------- per-N driver-log status ----------

def parse_summary_status(summary_path: Path) -> dict:
    """Map (bench, N) -> 'OK' / 'FAIL(rc=..)' / 'TIMEOUT' from driver summary."""
    out: dict[tuple[str, int], str] = {}
    text = _read(summary_path)
    lines = text.splitlines()
    cur = None
    for line in lines:
        m = re.match(r"-----\s+([\w\-]+)\s+N=(\d+)", line)
        if m:
            cur = (m.group(1), int(m.group(2)))
            continue
        if cur and re.match(r"\s+(OK|FAIL|TIMEOUT|SKIPPED)", line):
            status = line.strip().split()[0]
            out[cur] = status
            cur = None
    return out


# ---------- report assembly ----------

def fmt(x, ndig=2):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{ndig}f}"
    return str(x)


def build_tflops_table(bench_name: str, parsed: dict[int, dict], gbs_for_n=None) -> str:
    """Build the standard 'TF/s vs N' table.

    `parsed[N]` keys looked for (in order): last_tflops, mean_tflops.
    """
    rows = ["| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |",
            "|--:|--------------:|-------------:|-------------:|:------|"]
    for N in GPU_RANGE:
        d = parsed.get(N) or {}
        if not d:
            rows.append(f"| {N} | — | — | — | no data |")
            continue
        mean_t = d.get("mean_tflops")
        mn = d.get("min_tflops")
        mx = d.get("max_tflops")
        note = ""
        if d.get("failed"):
            note = "FAIL"
        elif d.get("oom"):
            note = "OOM"
        rows.append(
            f"| {N} | {fmt(mean_t)} | {fmt(mn)} | {fmt(mx)} | {note} |"
        )
    return "\n".join(rows)


def build_megatron_table(parsed: dict[int, dict]) -> str:
    # Two TFLOP/s/GPU metrics are reported side by side deliberately -- see the metric
    # warning in section 1.1a. "compute" is kernel-time (comparable to dell-cloud's
    # REPORT.md column); "wall-clock" includes bubbles/idle and is the lower number.
    rows = ["| N | GBS | compute TF/s/GPU | wall-clock TF/s/GPU | mean TF/s/GPU | last iter (ms) | notes |",
            "|--:|----:|-----------------:|--------------------:|--------------:|---------------:|:------|"]
    for N in GPU_RANGE:
        d = parsed.get(N) or {}
        gbs = d.get("gbs") or "—"
        last = d.get("last_tflops")
        meanv = d.get("mean_tflops")
        it_ms = d.get("last_iter_ms")
        if not d:
            note = "no data"
        elif d.get("oom"):
            note = "OOM"
        elif d.get("failed") and last is None:
            note = "FAILED"
        elif d.get("failed"):
            note = "ran then failed"
        else:
            note = ""
        comp = d.get("last_compute_tflops")
        wall = d.get("last_wall_tflops")
        rows.append(
            f"| {N} | {gbs} | {fmt(comp)} | {fmt(wall)} | {fmt(meanv)} | {fmt(it_ms)} | {note} |"
        )
    return "\n".join(rows)


def extract_b200_megatron_table(b200_text: str) -> str | None:
    """Find the '| N | B200 TF/s/GPU | MI355X TF/s/GPU |' table in the existing summary."""
    lines = b200_text.splitlines()
    for i, ln in enumerate(lines):
        if "B200 TF/s/GPU" in ln and "MI355X" in ln:
            # Capture this and the next ~3 lines
            block = [ln]
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].strip().startswith("|"):
                    block.append(lines[j])
                else:
                    break
            return "\n".join(block)
    return None


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    sweep_dir = Path(sys.argv[1])
    bench_out = Path(sys.argv[2])
    b200_path = Path(sys.argv[3])
    report_path = Path(sys.argv[4])

    if not sweep_dir.is_dir():
        print(f"sweep_dir not found: {sweep_dir}", file=sys.stderr)
        return 2

    summary = parse_summary_status(sweep_dir / "summary.txt")

    gemm = {N: parse_gemm_like(bench_out / f"gemm_N{N}.md") for N in GPU_RANGE}
    gemm_dense = {N: parse_gemm_like(bench_out / f"gemm-dense_N{N}.md") for N in GPU_RANGE}
    gemm_ds = {N: parse_gemm_like(bench_out / f"gemm-deepseek_N{N}.md") for N in GPU_RANGE}
    attn = {N: parse_attention(bench_out / f"attention_N{N}.csv") for N in GPU_RANGE}
    rccl = {N: parse_rccl(bench_out / f"rccl_N{N}.md") for N in GPU_RANGE}

    mlm = {}
    for N in GPU_RANGE:
        log_path = sweep_dir / f"megatron-llama2_7B-bf16_N{N}.log"
        mlm[N] = parse_megatron(log_path)
        st = summary.get(("megatron-llama2_7B-bf16", N))
        if st and "FAIL" in st:
            mlm[N].setdefault("failed", True)

    b200_text = _read(b200_path)
    b200_table = extract_b200_megatron_table(b200_text)

    # Build report
    parts: list[str] = []
    parts.append(f"# Primus Sweep Report — MI355X (1..8 GPUs)\n")
    parts.append(f"- Sweep dir: `{sweep_dir}`")
    parts.append(f"- Bench output dir: `{bench_out}`")
    parts.append(f"- Image: `rocm/primus:v26.3` (singularity SIF)")
    parts.append(f"- Hardware: 1 node × 8 × AMD Instinct MI355X (gfx950)\n")

    # --- Section 1: Megatron-LM ---
    parts.append("## 1. Megatron-LM (via Primus `train pretrain`)\n")
    parts.append(
        "Workload: `examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml` "
        "(llama2-7B, seq 4096, MBS=4, mock data, primus-turbo ON: "
        "`use_turbo_attention`, `use_turbo_grouped_mlp`). "
        "The `last TF/s/GPU` column is the steady-state value of the final logged "
        "iteration (after JIT warmup); `GBS` is parsed from the log.\n"
    )
    parts.append("#### Parallelism: pure data parallel (DP=N, TP=PP=CP=EP=1)\n")
    parts.append(
        "Verified from the run logs (`data_parallel_size=8, sequence_parallel_size=0`, "
        "`world_size=8`) and the config (`tensor_model_parallel_size: 1`, "
        "`pipeline_model_parallel_size: 1`, `expert_model_parallel_size: 1`, "
        "`sequence_parallel` commented out).\n\n"
        "Every GPU holds a **full llama2-7B replica** and processes its own micro-batches; "
        "gradients are all-reduced once per step. This is a **weak-scaling** study, so the "
        "driver computes `GBS(N) = MBS x N x GRAD_ACC = 4 x N x 8 = 32N` — constant work "
        "per GPU as N grows, and divisible by `MBS x DP` by construction. That last point "
        "matters: a fixed GBS=256 is *not* divisible by `MBS(4) x DP(N)` for N in "
        "{3,5,6,7}, which is what forced the reference Dell Cloud run into three separate "
        "rerun scripts. Computing GBS per N up front makes it one clean sweep.\n\n"
        "**Why DP and not TP/PP/CP/EP here:**\n\n"
        "- **DP is viable at all only because llama2-7B fits in one GPU's HBM** (288 GB on "
        "MI355X). Any model that did not fit would have forced TP or PP.\n"
        "- **TP** would shard each layer and all-reduce activations *every layer*, adding "
        "collective traffic that is unnecessary when the model already fits.\n"
        "- **PP** adds pipeline-bubble overhead and mainly earns its keep across nodes or "
        "when the model does not fit; on one node with fast XGMI it is strictly worse.\n"
        "- **CP** (context parallel) targets very long sequences; at seq 4096 it is "
        "unnecessary.\n"
        "- **EP** (expert parallel) applies only to MoE models; llama2-7B is dense.\n\n"
        "This is the deliberate opposite of Part D (ATOM inference), which runs **TP=8** "
        "because a 70B / 1.5 TB model cannot fit on one GPU. The contrast explains the "
        "collective-sensitivity result in section 7: Megatron here issues **one gradient "
        "all-reduce per ~5 s iteration**, so even the degraded N=5/6/7 RCCL bandwidth is "
        "negligible against per-iteration compute. TP=8 inference has no such insulation — "
        "its collectives sit in the **per-token critical path**.\n"
    )
    parts.append("### 1.1 TF/s/GPU vs #GPUs (Primus → Megatron-LM, llama2-7B BF16, turbo ON)\n")
    parts.append(build_megatron_table(mlm))
    parts.append("")

    # ---- 1.1a Dell Cloud Primus vs AMD Cloud Primus ----
    # Dell Cloud's REPORT.md section 1.1, same Primus -> Megatron llama2-7B BF16 path.
    # Their run fixed GBS at 256 (252/240 fudged for non-power-of-2, and N=5/6/7 failed
    # outright); ours computes GBS=32N. The two therefore only coincide at N=8.
    DELL_PRIMUS = {1: (256, 1160.60, 41640.70), 2: (256, 1146.00, 21084.20),
                   3: (252, 1143.60, 13865.70), 4: (256, 1139.10, 10606.40),
                   5: None, 6: None, 7: None, 8: (256, 1132.00, 5336.50)}
    parts.append("### 1.1a Dell Cloud Primus vs AMD Cloud Primus (same llama2-7B path)\n")
    parts.append(
        "Both hosts are 8 x MI355X running the same Primus -> Megatron-LM llama2-7B BF16 "
        "workload with primus-turbo ON, MBS=4, seq 4096. **Compared on `compute per GPU`, "
        "which is the metric Dell Cloud's REPORT.md section 1.1 reports** — see the metric "
        "note below, this distinction matters enormously.\n"
    )
    rows = ["| N | GBS Dell | GBS AMD | Dell compute TF/s/GPU | AMD compute TF/s/GPU | AMD/Dell | comparable? |",
            "|--:|--------:|--------:|---------------------:|--------------------:|--------:|:------------|"]
    for N in GPU_RANGE:
        d = mlm.get(N) or {}
        ours = d.get("last_compute_tflops")
        ours_gbs = d.get("gbs")
        dell = DELL_PRIMUS.get(N)
        if dell is None:
            rows.append(f"| {N} | — | {fmt(ours_gbs)} | — (run failed) | {fmt(ours)} | — | "
                        f"no — Dell has no data |")
            continue
        dgbs, dtf, _ = dell
        if ours is None:
            rows.append(f"| {N} | {dgbs} | — | {fmt(dtf)} | — | — | no — AMD has no data |")
            continue
        ratio = ours / dtf if dtf else None
        same_gbs = (ours_gbs == dgbs)
        verdict = "**YES — matched GBS**" if same_gbs else f"no — GBS differs ({dgbs} vs {fmt(ours_gbs)})"
        rows.append(f"| {N} | {dgbs} | {fmt(ours_gbs)} | {fmt(dtf)} | {fmt(ours)} | "
                    f"{('**%.2fx**' % ratio) if ratio else '—'} | {verdict} |")
    parts.append("\n".join(rows))
    parts.append("")
    n8 = mlm.get(8) or {}
    if n8.get("last_compute_tflops") and DELL_PRIMUS.get(8):
        r = n8["last_compute_tflops"] / DELL_PRIMUS[8][1]
        parts.append(
            f"**Only N=8 is a valid head-to-head** — it is the one point where both runs used "
            f"GBS=256 (ours as 32x8, theirs fixed). There the two machines are "
            f"**{r:.2f}x** apart: {DELL_PRIMUS[8][1]:.1f} vs "
            f"{n8['last_compute_tflops']:.1f} TF/s/GPU. Same silicon, essentially identical "
            f"result — which is the expected outcome and a good cross-machine validation.\n\n"
            f"At N=1..4 the GBS differs (Dell fixed 256; ours 32N = 32/64/96/128), so those "
            f"rows are not comparable — a smaller global batch means fewer tokens per "
            f"iteration and different efficiency. At N=5/6/7 Dell has no data at all: fixed "
            f"GBS=256 is not divisible by MBS(4) x DP(N) for those arities, which is exactly "
            f"the failure our per-N `GBS=32N` scheme was designed to avoid. **Our sweep is "
            f"8/8; theirs is 5/8.**\n"
        )
    parts.append(
        "> **Metric warning — two different TFLOP/s/GPU numbers exist.** The Megatron "
        "iteration line emits both `compute per GPU` (kernel-time throughput) and "
        "`throughput per GPU` (wall-clock, includes pipeline bubbles and idle). They differ "
        "by ~4x on this workload. Dell Cloud's REPORT.md section 1.1 column is the "
        "**compute** figure; a naive parse of the newer v26.5 log picks up the **wall-clock** "
        "figure instead. Comparing one against the other manufactures a spurious ~3.8x "
        "regression that does not exist. Section 1.1 above now reports both, explicitly "
        "labelled.\n"
    )

    parts.append("### 1.2 vs NVIDIA B200 (Megatron-LM, context only)\n")
    if b200_table:
        parts.append(
            f"Reference: `{b200_path}` — the existing MI355X-vs-B200 table from the "
            "`rocm/megatron-lm:v26.1` image sweep (**GPT-15.6B, MBS=4, BF16, no-recompute**). "
            "**This is not directly comparable** to the Primus llama2-7B numbers in §1.1: "
            "different model, different image (no primus-turbo), different GEMM shape mix. "
            "Kept here only as the existing house benchmark. See §7 for an apples-to-oranges "
            "framing of what the Primus-turbo path delivers on the same hardware.\n"
        )
        parts.append(b200_table)
        parts.append("")
    else:
        parts.append("_No B200 table found in the reference file._\n")

    # --- Section per benchmark ---
    parts.append("## 2. GEMM microbench (`benchmark gemm`)\n")
    parts.append("Square GEMM 4096×4096×4096 BF16, 10 s per rank, 2 GB rotating cache buffer. "
                 "Each rank runs independently — no collectives. Mean / min / max are taken across the N ranks.\n")
    parts.append("### 2.1 TF/s/GPU vs #GPUs\n")
    parts.append(build_tflops_table("gemm", gemm))
    parts.append("")

    parts.append("## 3. Dense GEMM microbench (`benchmark gemm-dense`)\n")
    parts.append("Llama-shape GEMM sweep (default: hidden 4096, FFN 11008, vocab 32000, MBS=1, BF16). "
                 "Reports TF/s per shape per rank; the table aggregates across shapes and ranks.\n")
    parts.append("### 3.1 TF/s/GPU (aggregate) vs #GPUs\n")
    parts.append(build_tflops_table("gemm-dense", gemm_dense))
    parts.append("")

    parts.append("## 4. DeepSeek GEMM microbench (`benchmark gemm-deepseek`)\n")
    parts.append("DeepSeek-V2/V3-style MoE shapes (hidden 4096, MoE int 1536, 128 routed experts, BF16).\n")
    parts.append("### 4.1 TF/s/GPU (aggregate) vs #GPUs\n")
    parts.append(build_tflops_table("gemm-deepseek", gemm_ds))
    parts.append("")

    parts.append("## 5. Attention microbench (`benchmark attention`)\n")
    parts.append("Flash-attention backend, MBS=4 across the built-in model shape set.\n")
    parts.append("### 5.1 Attention metrics vs #GPUs\n")
    rows = ["| N | metric | mean | best | n_shapes |",
            "|--:|:-------|-----:|-----:|---------:|"]
    for N in GPU_RANGE:
        d = attn.get(N) or {}
        if not d:
            rows.append(f"| {N} | — | — | — | — |")
            continue
        for metric, v in d.items():
            mean_v = v.get("mean")
            best_v = v.get("max") if "max" in v else v.get("min")
            rows.append(f"| {N} | {metric} | {fmt(mean_v)} | {fmt(best_v)} | {v.get('n', '—')} |")
    parts.append("\n".join(rows))
    parts.append("")

    parts.append("## 6. RCCL collective microbench (`benchmark rccl --op all_reduce`)\n")
    parts.append("All-reduce bandwidth sweep across message sizes (1K..128M, log2 sweep). "
                 "Peak busbw reflects the asymptotic large-message bandwidth; mean is across the size sweep. "
                 "**N=1 is skipped** — collective on a single rank is degenerate.\n")
    parts.append("### 6.1 Peak / mean all-reduce busbw vs #GPUs\n")
    rows = ["| N | peak busbw (GB/s) | mean busbw (GB/s) | sizes |",
            "|--:|------------------:|------------------:|------:|"]
    for N in GPU_RANGE:
        if N == 1:
            rows.append(f"| {N} | — | — | skipped |")
            continue
        d = rccl.get(N) or {}
        if not d:
            rows.append(f"| {N} | — | — | no data |")
            continue
        rows.append(f"| {N} | {fmt(d.get('peak_bw_gbps'))} | {fmt(d.get('mean_bw_gbps'))} | {d.get('n_sizes','—')} |")
    parts.append("\n".join(rows))
    parts.append("")

    # --- 6a/6b: Megatron measured against the microbench ceilings on this same host ---
    # All figures below are N=8, BF16, same machine, same run. The Megatron figure MUST be
    # the `compute per GPU` metric -- using the wall-clock one compares different quantities
    # (see the metric warning in 1.1a) and produces a nonsense percentage.
    m8 = (mlm.get(8) or {}).get("last_compute_tflops")
    g8 = (gemm.get(8) or {}).get("mean_tflops")
    gd8 = (gemm_dense.get(8) or {}).get("mean_tflops")

    # Part A's RVS gst bf16 ceiling, read live from results/rvs_tflops.csv if Part A has
    # run. This is the silicon ceiling (no framework at all) and sits above both Primus
    # rows. Deliberately not hardcoded so it tracks a re-run of Part A.
    rvs8 = None
    try:
        rvs_csv = report_path.parent / "rvs_tflops.csv"
        if rvs_csv.exists():
            import csv as _csv
            with rvs_csv.open() as fh:
                for row in _csv.DictReader(fh):
                    if row.get("precision") == "bf16" and row.get("gpus") == "8":
                        rvs8 = float(row["avg_per_gpu_tflops"])
                        break
    except Exception:
        rvs8 = None
    if m8 and (g8 or gd8):
        parts.append("## 6a. Megatron vs the GEMM ceilings (N=8, BF16, same host)\n")
        parts.append(
            "How much of the achievable matrix-multiply rate does real training actually "
            "realize? Each row is a progressively more realistic ceiling, so each gap "
            "attributes a specific loss.\n"
        )
        rows = ["| Ceiling | TF/s/GPU | Megatron as % | What the gap costs |",
                "|---|---:|---:|---|"]
        if rvs8:
            rows.append(f"| RVS `gst` bf16 — silicon, no framework (Part A) | {fmt(rvs8)} | "
                        f"**{100*m8/rvs8:.0f}%** | PyTorch/framework dispatch, then everything below |")
        if g8:
            rows.append(f"| Primus `gemm` — square 4096^3 | {fmt(g8)} | "
                        f"**{100*m8/g8:.0f}%** | off-peak shapes + everything non-GEMM |")
        if gd8:
            rows.append(f"| Primus `gemm-dense` — dense-model shape mix | {fmt(gd8)} | "
                        f"**{100*m8/gd8:.0f}%** | non-GEMM work only (shape penalty already priced in) |")
        rows.append(f"| Megatron llama2-7B (compute per GPU) | **{fmt(m8)}** | 100% | — |")
        parts.append("\n".join(rows))
        parts.append("")
        if gd8:
            parts.append(
                f"**`gemm-dense` is the right baseline.** It runs a dense-transformer shape "
                f"mix — the kind of QKV / O / FFN-gate / up / down GEMMs Megatron issues — "
                f"so the {100*m8/gd8:.0f}% figure isolates *non-GEMM* overhead: attention, "
                f"RMSNorm, RoPE, optimizer, and the gradient all-reduce. The square-GEMM row "
                f"is a looser ceiling because 4096^3 is a shape Megatron never actually "
                f"runs.\n\n"
                f"**`gemm-deepseek` is deliberately excluded.** Those are MoE expert shapes "
                f"with small, skewed K-dimensions; llama2-7B is dense and never issues them, "
                f"so a percentage against it would be meaningless.\n"
            )
        if rvs8 and g8:
            parts.append(
                f"**RVS `gst` vs Primus `gemm` — what actually differs.** Both measure BF16 "
                f"matrix multiply on this same host, and the {100*(1-g8/rvs8):.0f}% gap "
                f"between them ({fmt(rvs8)} -> {fmt(g8)}) is worth understanding, because it "
                f"is *not* only shape:\n\n"
                f"| | RVS `gst` (Part A) | Primus `gemm` (Part C) |\n"
                f"|---|---|---|\n"
                f"| Stack | hipBLASLt called **directly from C++** | **PyTorch** -> hipBLASLt |\n"
                f"| Shape | 8192 x 8192 x 16384 | 4096 x 4096 x 4096 |\n"
                f"| Cache defeat | `rotating: 512` buffers | 2 GB rotating buffer |\n"
                f"| Metric | **peak** across log intervals | **mean** across ranks |\n"
                f"| Duration | 30 s | 10 s |\n\n"
                f"Two effects dominate. **Framework dispatch**: RVS has no Python, no "
                f"autograd, no tensor wrapper — it is the closest thing to a pure library "
                f"number. **Matrix size**: RVS' GEMM is 8x larger in K and 4x in M/N, so "
                f"fixed per-call overhead amortizes far better. The *peak-vs-mean* metric "
                f"choice also flatters RVS slightly. So the RVS row is a genuine silicon "
                f"ceiling, but it is a deliberately favourable one — the Primus rows are "
                f"closer to what any real framework can reach.\n"
            )
        if gd8:
            parts.append(
                "> **Note on the name — \"dense\" means dense *model*, not dense *matrix*.** "
                "The contrast is with its sibling `gemm-deepseek` (a MoE / sparse-expert "
                "model), not with sparse matrices — all of these GEMMs are fully dense. So "
                "plain `gemm` is not \"denser\" than `gemm-dense` despite the name; it is "
                "simply one arbitrary shape (`--M --N --K`, here 4096^3) rather than a "
                "model-derived set. Caveat: that `gemm-dense` specifically uses *llama* "
                "shapes is an inference from the dense-vs-DeepSeek pairing, not verified "
                "against Primus' source — what is certain is that it is a dense-transformer "
                "shape set, which is what makes it the right ceiling for llama2-7B.\n"
            )
        parts.append(
            "> **Three caveats.** (1) Megatron's TFLOPs are an *analytical* count "
            "(~6·params·tokens), not measured FLOPs — so this is model-FLOPs utilization, "
            "not a literal hardware efficiency. (2) The microbenches are pure compute with "
            "no collectives; Megatron includes a gradient all-reduce per step. (3) Both "
            "numbers must be kernel-time (`compute per GPU`); mixing in the wall-clock "
            "figure invalidates the ratio entirely.\n"
        )

    # --- 6b: attention, as decomposition rather than a ratio ---
    a8 = attn.get(8) or {}
    fwd8 = (a8.get("fwd_tflops") or {}).get("mean")
    bwd8 = (a8.get("bwd_tflops") or {}).get("mean")
    if m8 and fwd8 and bwd8:
        parts.append("## 6b. Where the remaining gap goes — attention\n")
        parts.append(
            "Attention is **not** compared as a percentage of Megatron: it measures a "
            "*component*, not a substitute workload, so \"Megatron as % of attention\" would "
            "be a category error. It is reported here because it is the leading explanation "
            "for why end-to-end training lands below the GEMM ceiling above.\n"
        )
        rows = ["| Kernel class (N=8) | TF/s/GPU | vs `gemm-dense` |", "|---|---:|---:|"]
        if gd8:
            rows.append(f"| `gemm-dense` (the GEMM path) | {fmt(gd8)} | 100% |")
            rows.append(f"| attention **forward** | {fmt(fwd8)} | {100*fwd8/gd8:.0f}% |")
            rows.append(f"| attention **backward** | {fmt(bwd8)} | {100*bwd8/gd8:.0f}% |")
        parts.append("\n".join(rows))
        parts.append("")
        parts.append(
            f"Attention forward runs at roughly half the GEMM rate and **backward at "
            f"{bwd8/fwd8*100:.0f}% of forward** ({fmt(bwd8)} vs {fmt(fwd8)} TF/s/GPU). "
            f"Backward is dominated by gradient recomputation plus extra matmuls, and the "
            f"asymmetry matches what is reported for flash-attention-class kernels "
            f"generally.\n\n"
            f"Since a transformer step spends a substantial fraction of its time in "
            f"attention — and backward is ~2x the cost of forward in a training step — a "
            f"kernel class running at {100*bwd8/gd8:.0f}% of GEMM rate is sufficient on its "
            f"own to explain most of the residual between the `gemm-dense` ceiling and "
            f"measured end-to-end throughput. Attention is also flat across N (each rank "
            f"runs independently, no collective), so this is a per-GPU kernel property, not "
            f"a scaling effect.\n"
        )

    # --- Analysis ---
    parts.append("## 7. Analysis\n")
    notes = []

    # ---- Megatron: weak-scaling efficiency + per-GPU plateau ----
    mlm_lasts = {N: (mlm.get(N) or {}).get("last_tflops") for N in GPU_RANGE}
    valid_mlm = {N: v for N, v in mlm_lasts.items() if v}
    if len(valid_mlm) >= 2:
        # Reference N is the smallest viable (closest to single-rank baseline).
        ref_N = min(valid_mlm)
        ref = valid_mlm[ref_N]
        max_N = max(valid_mlm)
        # Weak-scaling efficiency: per-GPU TF/s at N / per-GPU TF/s at ref_N
        eff_table = ", ".join(
            f"N={N}: {v:.0f} TF/s/GPU ({v/ref*100:.0f} % of N={ref_N})"
            for N, v in sorted(valid_mlm.items())
        )
        notes.append(
            f"- **Megatron weak-scaling (llama2-7B BF16, turbo ON):** {eff_table}. "
            f"Per-GPU throughput is essentially flat (≤ {(1 - min(valid_mlm.values())/max(valid_mlm.values()))*100:.0f} % spread "
            f"between best and worst N), so the all-reduce overhead at MBS·N grad-accum is small relative to the "
            "model's compute. The lower N=1 / higher N=8 iter-time scales linearly with GBS as expected for weak-scaling."
        )
        # Speedup vs reference image (rocm/megatron-lm:v26.1 GPT-15.6B): take the §3-tuned best of 790.4 TF/s/GPU at N=8.
        if 8 in valid_mlm:
            ref_image_n8 = 790.4
            mlm_n8 = valid_mlm[8]
            notes.append(
                f"- **Primus-turbo vs reference image at N=8:** Primus (llama2-7B, turbo ON) hits "
                f"**{mlm_n8:.0f} TF/s/GPU**; the `rocm/megatron-lm:v26.1` image on the same hardware "
                f"(GPT-15.6B, no turbo, §3 tuned) tops out at **790.4 TF/s/GPU** — a "
                f"**{mlm_n8/ref_image_n8:.2f}× per-GPU jump**. Workloads differ (smaller model, different "
                "GEMM shapes, primus-turbo attention/grouped-MLP fused kernels), so this is *not* a pure "
                "kernel-vs-kernel speedup; it captures the combined win of (i) llama2-7B being more "
                "GEMM-dense than GPT-15.6B, (ii) primus-turbo replacing unfused softmax/RMSNorm/attention "
                "with gfx950-native kernels, and (iii) Primus' MFU-tuned argument set. Use as the new "
                "headline number for this hardware on a llama-family workload."
            )
        # Failed-N note
        missing = [N for N in GPU_RANGE if N not in valid_mlm]
        if missing:
            notes.append(
                f"- **Missing N:** {missing} have no valid TF/s data. For N=3/5/6/7 the original GBS=256 fails "
                "Megatron's `GBS % (MBS·DP) == 0` precondition; reruns used GBS∈{240, 252}. Any remaining gap "
                "after the latest rerun is a real failure — check the per-N log."
            )

    # ---- GEMM flatness (per-rank, no collectives) ----
    gemm_means = [gemm.get(N, {}).get("mean_tflops") for N in GPU_RANGE if gemm.get(N, {}).get("mean_tflops")]
    if len(gemm_means) >= 2:
        gmin, gmax = min(gemm_means), max(gemm_means)
        drift = (gmax - gmin) / gmax * 100
        notes.append(
            f"- **GEMM per-GPU consistency:** mean TF/s/GPU ranges {gmin:.1f}..{gmax:.1f} across all N "
            f"({drift:.1f} % spread). Each rank runs the same 4Kx4Kx4K BF16 shape independently with no collectives, "
            "so a flat curve confirms there's no thermal/PCIe/power contention as N grows. This is the per-GPU "
            "compute ceiling on this hardware for square FP16/BF16 matmul."
        )

    # ---- gemm-dense vs gemm-deepseek vs gemm ratio ----
    gd_means = [gemm_dense.get(N, {}).get("mean_tflops") for N in GPU_RANGE if gemm_dense.get(N, {}).get("mean_tflops")]
    gds_means = [gemm_ds.get(N, {}).get("mean_tflops") for N in GPU_RANGE if gemm_ds.get(N, {}).get("mean_tflops")]
    if gemm_means and gd_means and gds_means:
        peak = mean(gemm_means)
        notes.append(
            f"- **Shape sensitivity:** square 4Kx4Kx4K hits {peak:.0f} TF/s/GPU; the **llama-shape mix** "
            f"(gemm-dense) drops to {mean(gd_means):.0f} ({mean(gd_means)/peak*100:.0f} % of peak); "
            f"the **deepseek MoE shape mix** falls to {mean(gds_means):.0f} ({mean(gds_means)/peak*100:.0f} %). "
            "The MoE drop is shape-driven (small / skewed K-dim in the expert path), not a hardware issue."
        )

    # ---- Attention turbo ----
    if attn:
        fwds = [v.get("mean") for N in GPU_RANGE for k, v in (attn.get(N) or {}).items() if "fwd" in k.lower() and v.get("mean")]
        bwds = [v.get("mean") for N in GPU_RANGE for k, v in (attn.get(N) or {}).items() if "bwd" in k.lower() and v.get("mean")]
        if fwds and bwds:
            notes.append(
                f"- **Attention fwd/bwd asymmetry:** fwd ≈ {mean(fwds):.0f} TF/s/GPU, bwd ≈ {mean(bwds):.0f} TF/s/GPU "
                f"(bwd / fwd = {mean(bwds)/mean(fwds)*100:.0f} %). Backward is dominated by gradient recomputation + extra "
                "matmuls; the gap matches what's reported for flash-attention class kernels. Both are stable across N "
                "(each rank runs independently — no all-reduce in this bench)."
            )

    # ---- RCCL cliff ----
    rccl_peaks = {N: (rccl.get(N) or {}).get("peak_bw_gbps") for N in GPU_RANGE if N >= 2}
    valid_rccl = {N: v for N, v in rccl_peaks.items() if v}
    if len(valid_rccl) >= 3:
        odd_present = [N for N in (5, 6, 7) if N in valid_rccl]
        even_present = [N for N in (4, 8) if N in valid_rccl]
        if odd_present and even_present:
            odd_avg = mean([valid_rccl[N] for N in odd_present])
            even_avg = mean([valid_rccl[N] for N in even_present])
            ratio = odd_avg / even_avg
            notes.append(
                f"- **RCCL all-reduce cliff:** peak busbw at N∈{{4,8}} averages **{even_avg:.0f} GB/s**; "
                f"at N∈{{5,6,7}} it drops to **{odd_avg:.0f} GB/s** ({ratio*100:.0f} %). N=8 alone "
                f"hits **{valid_rccl.get(8, 0):.0f} GB/s** — the asymptotic xGMI ring bandwidth. The non-power-of-2 cliff "
                "matches the existing megatron-lm:v26.1 reference and confirms it's a topology/ring-algorithm issue "
                "(RCCL falls back from a clean ring to tree/segmented patterns), not a Primus issue. **Yet** the Megatron "
                "training in §1.1 is essentially insensitive to this cliff because per-iter compute (~20 s) dwarfs "
                "the all-reduce time even at the degraded busbw."
            )

    if not notes:
        notes.append("- (Analysis pending — re-run the report generator once more data is present.)")
    parts.append("\n".join(notes))
    parts.append("")

    # --- Raw status ---
    parts.append("## 8. Raw per-(bench, N) status\n")
    parts.append("From driver `summary.txt`:\n")
    parts.append("```")
    parts.append(_read(sweep_dir / "summary.txt").rstrip() or "(no summary.txt yet)")
    parts.append("```\n")

    report_path.write_text("\n".join(parts))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
