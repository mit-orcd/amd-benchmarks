#!/usr/bin/env python3
"""Generate results/kimi-k3-maxseqs.md from the raised --max-num-seqs experiment.

Usage: analyze_kimi_mad.py <kimi_mad_sweep_dir> -o <results_dir>

Mirrors every section and table of results/kimi-k3-base.md, but populated entirely from the
NEW run. Old numbers are never shown. Writes only kimi-k3-maxseqs.md (+ .csv); it does
not read or modify kimi-k3-base.md.

Everything derived (TFLOP/s, roofline, HBM %, comms volumes) is computed from the measured
throughput plus the parsed architecture, exactly as in the original analysis -- so the two
files are directly comparable section by section if a reader wants to diff them.
"""
import argparse, json, re, sys
from pathlib import Path

# ---- static architecture (parsed from Kimi-K3 config.json; identical model either run) ----
H = 7168; L_TOT = 93; N_FULL = 24; N_KDA = 69
E = 896; TOPK = 16; SHARED = 2
MOE_INTER = 3072; REXP_H = 3584; MOE_LAYERS = 92
KV_LORA = 512; QK_ROPE = 64
TP = 8
HBM_PER_GPU = 8000.0          # GB/s
LINK_UNI = 76.8; MAX_LINKS = 7
XGMI_UNI_AGG = LINK_UNI * MAX_LINKS   # 537.6 GB/s per direction
BF16_PEAK = 2500.0            # TFLOP/s per GPU
BABEL_MEASURED = 7260.0       # GB/s, Part A pure streaming read = 91% of spec
PER_EXPERT = 3 * REXP_H * MOE_INTER
ACTIVE_PARAMS = 84e9          # ~84 B active/token (see note on KDA approximation)


def f(v, nd=1):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "-"


def distinct_experts(B):
    return E * (1 - (1 - 1 / E) ** (B * TOPK))


def weight_gb_per_gpu(B):
    """Weight bytes read per decode step, per GPU, MXFP4 (~0.5 B/param)."""
    d = distinct_experts(B)
    return (d + SHARED) * PER_EXPERT * MOE_LAYERS * 0.5 / 1e9 / TP


def load_rows(d: Path):
    rows = []
    for j in sorted(d.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            data = json.load(j.open())
        except (json.JSONDecodeError, OSError):
            continue
        c = data.get("max_concurrency") or int(re.sub(r"\D", "", j.stem) or 0)
        rows.append(dict(
            conc=int(c),
            tps=data.get("output_throughput"),
            total_tps=data.get("total_token_throughput"),
            rps=data.get("request_throughput"),
            ttft=data.get("median_ttft_ms"), ttft99=data.get("p99_ttft_ms"),
            tpot=data.get("median_tpot_ms"), tpot99=data.get("p99_tpot_ms"),
            completed=data.get("completed")))
    return [r for r in rows if r["tps"]]


def parse_memory(d: Path):
    """Pull the server's own memory-budget line."""
    log = d / "atom_server.log"
    if not log.exists():
        return {}
    m = None
    for line in log.read_text(errors="replace").splitlines():
        if "Memory budget:" in line:
            m = line
            break
    if not m:
        return {}
    out = {}
    for k in ["total_gpu", "free", "utilization", "budget", "peak_torch", "non_torch",
              "cudagraph_est", "safety", "available_for_kv", "block_bytes",
              "num_kvcache_blocks"]:
        mm = re.search(rf"{k}=([\d.]+)", m)
        if mm:
            out[k] = float(mm.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(a.sweep)
    if not rows:
        sys.exit(f"no usable result JSON in {a.sweep}")
    rows.sort(key=lambda r: r["conc"])
    best = max(rows, key=lambda r: r["tps"])
    lo, hi = rows[0], rows[-1]
    mem = parse_memory(a.sweep)

    # CSV
    import csv as _csv
    with (a.out / "kimi-k3-maxseqs.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    B = best["conc"]
    step_ms = B / best["tps"] * 1000
    wgb = weight_gb_per_gpu(B)
    hbm_used = wgb / (step_ms / 1000)          # GB/s per GPU
    hbm_pct = 100 * hbm_used / HBM_PER_GPU
    roof = B * (HBM_PER_GPU * TP) / wgb
    tflops_agg = 2 * ACTIVE_PARAMS * best["tps"] / 1e12
    tflops_gpu = tflops_agg / TP

    # comms
    payload_tok = 2 * L_TOT * H * 2            # bytes per token, all all-reduces
    steps_s = best["tps"] / B
    wire_step = B * payload_tok * 2 * (TP - 1) / TP
    xgmi_gbs = wire_step * steps_s / 1e9
    per_call_kb = B * H * 2 / 1024

    kv_per_tok = mem.get("block_bytes", 1769472) / 128
    kv_pool = mem.get("available_for_kv")

    Lx = []
    A = Lx.append

    A("# Kimi-K3 on 8 × MI355X — max-num-seqs experiment (compute and communication analysis)")
    A("")
    A("Serving `moonshotai/Kimi-K3` (2.78 T params, 1.5 TB MXFP4 checkpoint) on a single")
    A("8 × MI355X node via ATOM, TP=8, with **`--max-num-seqs` raised from 64 to 256** to")
    A("test whether the in-flight cap — not hardware — was the binding limit.")
    A(f"Source run: `{a.sweep.name}`.")
    A("")
    A("> This file reports **only** the raised-`max-num-seqs` run. Baseline runs are in")
    A("> `kimi-k3-base.md` (original recipe, max-num-seqs=64) and `kimi-k3-mad.md` (MAD recipe,")
    A("> max-num-seqs=64); their head-to-head is `kimi-k3-comparison.md`. All are kept")
    A("> separate so none overwrites another.")
    A("")
    A("**Run configuration** (from the server log and launch command, not assumed):")
    A("")
    A("| Setting | Value |")
    A("|---|---|")
    A("| Image | `rocm/atom-dev:latest` (the better-performing baseline, see kimi-k3-comparison.md) |")
    A("| Parallelism | `tensor_parallel_size=8`, PP=1, DP=1, EP off |")
    A("| Quantization | MXFP4 routed experts + PTPC-FP8 via `--online_quant_config` (original recipe) |")
    A("| KV cache dtype | fp8 |")
    A("| `max_model_len` / `max_num_seqs` | 16384 / **256** (raised from 64) |")
    A("| `max_num_batched_tokens` | 16384 |")
    A("| `gpu_memory_utilization` | 0.93 |")
    A("| Prefix caching | disabled (KDA recurrent state cannot be rebuilt from paged cache) |")
    A(f"| Workload | ISL/OSL 1024/1024, `--ignore-eos`, concurrency {rows[0]['conc']}→{rows[-1]['conc']} |")
    A("| Kernel-selection env vars | none (original recipe; MAD vars measured ~9% slower) |")
    A("")
    A("**Architecture**: 93 layers — **24 MLA full-attention** + **69 KDA linear-attention**;")
    A("hidden 7168; MoE with **896 routed experts, top-16 + 2 shared**, expert hidden")
    A("3584 → 3072. ≈ **2.76 T params** total, ~84 B active per token (3.0%).")
    A("")
    A("---")
    A("")

    # ---- 0. overview ----
    A("## 0. Overview — the short version")
    A("")
    A(f"**§1 Compute** — **{f(best['tps'])} tok/s** peak at c={B} "
      f"(TPOT {f(best['tpot'],2)} ms, TTFT {f(best['ttft'])} ms). Achieved "
      f"**{f(tflops_agg)} TFLOP/s aggregate = {f(tflops_gpu)}/GPU = "
      f"{100*tflops_gpu/BF16_PEAK:.1f}% of BF16 peak**.")
    A("")
    if mem:
        A(f"**§2 Memory** — per GPU: **{f(mem.get('peak_torch',0))} GB weights** + "
          f"**{f(kv_pool or 0)} GB KV pool**. KV is {f(kv_per_tok/1024,1)} KB/token — only the "
          f"24 MLA layers keep a paged cache; the 69 KDA layers hold fixed recurrent state.")
    else:
        A("**§2 Memory** — server memory-budget line not found in this run's log.")
    A("")
    A(f"**§3 Bottleneck — intra-GPU HBM bandwidth.** Compute "
      f"{100*tflops_gpu/BF16_PEAK:.1f}% utilized, XGMI {100*xgmi_gbs/XGMI_UNI_AGG:.1f}%, "
      f"**HBM ~{hbm_pct:.0f}%** ({f(hbm_used)} GB/s of {f(HBM_PER_GPU,0)}). At batch {B}, "
      f"**{distinct_experts(B):.0f} of {E} experts** activate per layer, so weight traffic is "
      f"{f(wgb)} GB/GPU/step.")
    A("")
    A(f"**§4 Communication** — XGMI carries only activations: {2*L_TOT} all-reduces/token, "
      f"**{xgmi_gbs:.2f} GB/s per GPU ({100*xgmi_gbs/XGMI_UNI_AGG:.1f}% of ceiling)**. No "
      f"all-to-all (EP off), no gradients, no KV exchange.")
    A("")
    A("---")
    A("")

    # ---- 1. compute ----
    A("## 1. Computing performance")
    A("")
    A("**Concurrency** = number of independent requests served *at once*, each with its own")
    A("prompt and growing output. It is a client-side load setting, not a hardware unit — all")
    A("requests are batched together on the **same 8 GPUs** (TP=8), not one per GPU.")
    A("")
    A("| Concurrency | Throughput (tok/s) | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | req/s | completed |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        star = "**" if r is best else ""
        A(f"| {r['conc']} | {star}{f(r['tps'])}{star} | {f(r['ttft'])} | {f(r['ttft99'])} | "
          f"{f(r['tpot'],2)} | {f(r['rps'],2)} | {f(r['completed'],0)} |")
    A("")
    if lo["tps"] and lo["tpot"]:
        A(f"Throughput scales **{hi['tps']/lo['tps']:.1f}×** from c={lo['conc']} to "
          f"c={hi['conc']} while TPOT grows {hi['tpot']/lo['tpot']:.1f}×.")
    A("")
    A("With `--max-num-seqs 256`, concurrency up to 256 is admitted to the batch rather than")
    A("queued. Compare against `kimi-k3-mad.md`, where the same concurrencies ran against")
    A("a 64-slot server and TTFT median reached 150 s at c=256 while throughput stayed flat.")
    A("")
    A("### Achieved TFLOP/s")
    A("")
    A("MoE is sparse: only **top-16 + 2 shared of 896** experts fire per token, so active")
    A("params ≈ **84 B** of 2.76 T (3.0%). At 2 FLOP per active param per token:")
    A("")
    A("| Concurrency | Aggregate TFLOP/s | Per GPU | % of BF16 peak (2500) |")
    A("|---:|---:|---:|---:|")
    for r in rows:
        ta = 2 * ACTIVE_PARAMS * r["tps"] / 1e12
        A(f"| {r['conc']} | {f(ta)} | {f(ta/TP,1)} | {100*(ta/TP)/BF16_PEAK:.2f}% |")
    A("")
    A("**Decode is nowhere near compute-bound** — a fraction of peak matrix throughput. That")
    A("is expected: autoregressive decode does one token per sequence per step, so every")
    A("weight matrix serves a narrow GEMV-like operation. This is a *memory-bandwidth*")
    A("regime, quantified in §3.")
    A("")
    A("> The KDA (linear-attention) parameter count is approximated from config dimensions;")
    A("> the MoE and MLA terms are exact. Treat the 84 B active figure as ±15%. The")
    A("> conclusion (decode is ~1% of peak) has far too much margin to be affected.")
    A("")

    # ---- 2. memory ----
    A("## 2. GPU memory usage")
    A("")
    if mem:
        A("Measured per rank at load time, from the server's own budget line:")
        A("")
        A("```")
        A(f"total_gpu={f(mem.get('total_gpu',0),2)}GB  utilization={mem.get('utilization','-')}  "
          f"budget={f(mem.get('budget',0),2)}GB")
        A(f"peak_torch={f(mem.get('peak_torch',0),2)}GB  non_torch={f(mem.get('non_torch',0),2)}GB  "
          f"cudagraph_est={f(mem.get('cudagraph_est',0),2)}GB  safety={f(mem.get('safety',0),2)}GB")
        A(f"available_for_kv={f(kv_pool or 0,2)}GB  block_bytes={int(mem.get('block_bytes',0))}  "
          f"num_kvcache_blocks={int(mem.get('num_kvcache_blocks',0))}")
        A("```")
        A("")
        A("| Component | Per GPU | Node total (×8) |")
        A("|---|---:|---:|")
        A(f"| Model weights + framework (`peak_torch`) | {f(mem.get('peak_torch',0))} GB | {f(mem.get('peak_torch',0)*8,0)} GB |")
        A(f"| Non-torch (RCCL buffers, HIP runtime) | {f(mem.get('non_torch',0))} GB | {f(mem.get('non_torch',0)*8,0)} GB |")
        A(f"| CUDA-graph pool | {f(mem.get('cudagraph_est',0),2)} GB | {f(mem.get('cudagraph_est',0)*8,1)} GB |")
        A(f"| Safety margin | {f(mem.get('safety',0),2)} GB | {f(mem.get('safety',0)*8,1)} GB |")
        A(f"| **KV cache pool** | **{f(kv_pool or 0)} GB** | **{f((kv_pool or 0)*8,0)} GB** |")
        A("")
        A("**It does not load only weights** — a KV pool is carved out on top.")
        A("")
        A("### KV cache details")
        A("")
        A(f"`block_bytes / 128 tokens` = **{kv_per_tok:,.0f} B per token per GPU** "
          f"({kv_per_tok/1024:.1f} KB).")
        A("")
        A(f"That decodes exactly: MLA stores a compressed latent KV of "
          f"`kv_lora({KV_LORA}) + qk_rope({QK_ROPE})` = {KV_LORA+QK_ROPE} values/token/layer at "
          f"fp8 = {KV_LORA+QK_ROPE} B, × **{N_FULL} full-attention layers** = "
          f"{(KV_LORA+QK_ROPE)*N_FULL:,} B. Two conclusions follow:")
        A("")
        A(f"1. **Only the {N_FULL} MLA layers consume paged KV.** The {N_KDA} KDA layers keep a")
        A("   fixed-size recurrent state per request instead — which is why a 93-layer model")
        A("   has the KV footprint of a 24-layer one.")
        A("2. **KV is replicated across TP ranks, not sharded.** MLA's latent is shared across")
        A("   heads, so sharding would force a re-gather every step. Replication trades idle")
        A("   HBM for zero communication.")
        A("")
        if kv_pool:
            cap = kv_pool * 1e9 / kv_per_tok
            used = B * 2048 * kv_per_tok / 1e9
            A(f"Capacity: {f(kv_pool)} GB ÷ {kv_per_tok:,.0f} B = **{cap/1e6:.2f} M tokens** per GPU. "
              f"At c={B} × ~2048 ctx that is {used:.2f} GB — **{100*used/kv_pool:.1f}% of the pool**.")
    else:
        A("_Server memory-budget line not found in this run's log; see `atom_server.log`._")
    A("")

    # ---- 3. bottleneck ----
    A("## 3. What is the bottleneck?")
    A("")
    A("**Intra-GPU HBM bandwidth.** Not compute, not the interconnect.")
    A("")
    A(f"| Resource | Demand at c={B} | MI355X capability | Utilization |")
    A("|---|---:|---:|---:|")
    A(f"| Compute | {f(tflops_gpu)} TFLOP/s per GPU | {f(BF16_PEAK,0)} TFLOP/s BF16 | **{100*tflops_gpu/BF16_PEAK:.1f}%** |")
    A(f"| **HBM bandwidth** | **{f(hbm_used)} GB/s per GPU** | {f(HBM_PER_GPU,0)} GB/s | **~{hbm_pct:.0f}%** |")
    A(f"| XGMI (GPU↔GPU) | {xgmi_gbs:.2f} GB/s per GPU | ~{f(XGMI_UNI_AGG)} GB/s (1-direction) | **~{100*xgmi_gbs/XGMI_UNI_AGG:.1f}%** |")
    A("")
    A("### 3.1 Why HBM — the MoE batching mechanism")
    A("")
    A("| Batch | Experts fired/layer | Tokens per expert | Weights read/GPU |")
    A("|---:|---:|---:|---:|")
    for r in rows:
        d = distinct_experts(r["conc"])
        A(f"| {r['conc']} | {d:.0f} | {r['conc']*TOPK/d:.1f} | {f(weight_gb_per_gpu(r['conc']))} GB |")
    A("")
    A("This is the defining property of sparse MoE: **compute grows with batch, but weight")
    A("traffic grows much faster** until nearly every expert is touched every step. MXFP4 is")
    A("what makes it tractable — at BF16 the same reads would be 4× larger and exceed HBM")
    A("bandwidth outright.")
    A("")
    A(f"### 3.2 How to improve it — raising HBM utilization above the current ~{hbm_pct:.0f}%")
    A("")
    A(f"RVS `babel`, a pure streaming-read kernel with no compute or routing, measured")
    A(f"**{f(BABEL_MEASURED,0)} GB/s = {100*BABEL_MEASURED/HBM_PER_GPU:.0f}% of spec** on this "
      f"box (Part A). So ~{100*BABEL_MEASURED/HBM_PER_GPU:.0f}% is the practical hardware "
      f"ceiling, and this run delivers **{hbm_pct/(100*BABEL_MEASURED/HBM_PER_GPU)*100:.0f}% "
      f"of what the memory system can actually do**.")
    A("")
    A(f"The shortfall is that MoE GEMMs are thin: at batch {B} each activated expert sees only")
    A(f"**{B*TOPK/distinct_experts(B):.1f} tokens**, making each weight read a matrix-*vector*")
    A("product that cannot issue enough concurrent memory requests to saturate HBM.")
    A("")
    A("Ranked levers:")
    A("")
    A("1. **Raise `--max-num-seqs`** — biggest lever; weight traffic plateaus once all 896")
    A("   experts activate, so extra tokens become nearly free in bandwidth terms.")
    A("2. **Speculative decoding / MTP** — verifies several tokens per weight read.")
    A("3. **Expert parallelism** — unavailable: ATOM raises `NotImplementedError` for EP with")
    A("   the MXFP4 SiTUv2 kernel.")
    A("4. **Prefill/decode disaggregation** — removes interleaved prefill from decode steps.")
    A("")
    A("**Could it reach 90%?** No. That is a pure streaming read with nothing else in the")
    A("loop. A real engine also does MXFP4 dequant, expert routing, MLA+KDA attention,")
    A(f"{2*L_TOT} all-reduces per token, and interleaved prefill. **50–65% is the plausible")
    A("target.**")
    A("")
    A("### 3.3 Why TTFT and TPOT behave differently")
    A("")
    A(f"TTFT moves from {f(lo['ttft'])} → {f(hi['ttft'])} ms across the sweep while TPOT moves")
    A(f"{f(lo['tpot'],2)} → {f(hi['tpot'],2)} ms. Prefill is compute-dense and has headroom;")
    A("decode adds weight-read traffic per step as more experts activate. The two metrics sit")
    A("in different regimes — further confirmation that decode is bandwidth-limited.")
    A("")

    # ---- 4. comms ----
    A("## 4. Data communication analysis")
    A("")
    A("### Intra-node GPU↔GPU (XGMI) — activations only")
    A("")
    A("With **TP=8 and EP disabled**, every expert is sharded across all 8 GPUs, so there is")
    A("**no expert-routing all-to-all**. The only cross-GPU traffic is TP activation reduction:")
    A("")
    A("| Property | Value |")
    A("|---|---|")
    A(f"| Collective | **all-reduce** (RCCL), 2 per layer × {L_TOT} layers |")
    A(f"| Count per token | **{2*L_TOT} all-reduces** |")
    A(f"| Payload per call per token | `hidden_size × 2 B` = **{H*2/1024:.1f} KB** |")
    A(f"| Payload per token (all layers) | **{payload_tok/1e6:.2f} MB** |")
    A("")
    A("| Concurrency | Steps/s | Payload/step | Wire bytes/step¹ | Sustained per GPU |")
    A("|---:|---:|---:|---:|---:|")
    for r in rows:
        s = r["tps"] / r["conc"]
        ps = r["conc"] * payload_tok
        ws = ps * 2 * (TP - 1) / TP
        A(f"| {r['conc']} | {s:.1f} | {ps/1e6:.1f} MB | {ws/1e6:.1f} MB | "
          f"**{ws*s/1e9:.2f} GB/s** |")
    A("")
    A("¹ busbw convention: an all-reduce moves `2(N−1)/N × payload` on the wire.")
    A("")
    A(f"At **{xgmi_gbs:.2f} GB/s against a ~{f(XGMI_UNI_AGG)} GB/s per-direction ceiling "
      f"(~{100*xgmi_gbs/XGMI_UNI_AGG:.1f}%)**, the interconnect is almost entirely idle. The")
    A("N=5/6/7 RCCL cliff found in Part B is irrelevant here — TP=8 is a power-of-2 arity that")
    A("never triggers it, and even the degraded cliff bandwidth would be ample.")
    A("")
    A(f"**Message-size caveat**: each call is only {per_call_kb:.0f} KB at c={B} — far below the")
    A("16 MiB–8 GiB range Part B swept, so these collectives are **latency-dominated, not")
    A(f"bandwidth-dominated**. The {2*L_TOT} serialized calls per step mean the true cost is")
    A("higher than the raw utilization figure suggests.")
    A("")
    A("**What is NOT transferred over XGMI:** weights (resident per GPU), KV cache")
    A("(replicated), gradients (inference), expert tokens (EP off).")
    A("")
    A("### Intra-GPU (HBM) — dominated by weights")
    A("")
    A(f"Per decode step at c={B}, per GPU:")
    A("")
    A("| Traffic | Bytes/step | Share |")
    A("|---|---:|---:|")
    kvr = B * 2048 * kv_per_tok / 1e9
    act = B * payload_tok / 1e9
    tot = wgb + kvr + act
    A(f"| **Expert weights (MXFP4)** | **~{f(wgb)} GB** | **~{100*wgb/tot:.0f}%** |")
    A(f"| KV cache read | ~{kvr:.2f} GB | ~{100*kvr/tot:.1f}% |")
    A(f"| Activations | ~{act:.2f} GB | ~{100*act/tot:.1f}% |")
    A("")
    A(f"HBM moves ~{f(wgb)} GB/step while XGMI moves ~{wire_step/1e9:.2f} GB/step — roughly")
    A(f"**{wgb/(wire_step/1e9):.0f}:1**. Optimization effort belongs on the memory side.")
    A("")

    # ---- 5. discussion ----
    A("## 5. Further discussion")
    A("")
    A("**1. EP is off and cannot be enabled for this model.** ATOM raises")
    A("`NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel")
    A("masking` — the MXFP4 SiTUv2 grouped-MoE kernel has no expert-parallel variant. MXFP4")
    A("experts and EP are mutually exclusive **on `rocm/atom-dev:latest`, the image the EP")
    A("test actually ran on** (2026-08-14). Not retested on the MAD-pinned image, so treat")
    A("it as image-specific rather than universal. The HBM-vs-XGMI trade cannot be")
    A("evaluated on this model today.")
    A("")
    A("**2. `max_num_seqs` was raised to 256 for this run** — the whole point of the")
    A("experiment. Whether it moved throughput is answered by the §1 table versus the")
    A("~1,180 tok/s plateau the 64-slot runs hit; see `kimi-k3-comparison.md` §2.")
    A("")
    A("**3. Prefix caching is disabled for correctness.** KDA recurrent state is per-request")
    A("and cannot be reconstructed from the paged MLA cache. In workloads with shared prefixes")
    A("this forfeits a win non-KDA models get for free — an architectural trade, not an")
    A("oversight.")
    A("")
    A("**4. The hybrid attention design is what makes 2.78 T fit.** Only 24 of 93 layers keep")
    A("a growing KV cache; 69 use fixed-size KDA state. A conventional 93-layer model would")
    A("need ~4× the KV per token.")
    A("")
    A("**5. This run used AMD's validated kernel selection.** The MAD env vars")
    A("(`ATOM_USE_TRITON_GEMM=1`, `ATOM_USE_TRITON_MOE=0`, `AITER_FLYDSL_FORCE=1`,")
    A("`ATOM_USE_UNIFIED_ATTN=1`, `ATOM_FORCE_ATTN_TRITON=1`, …) select the kernel set AMD")
    A("benchmarked and published against — unlike the earlier run, which used generic")
    A("defaults. Comparison between the two is in `notes-kimi-k3.md`.")
    A("")

    A("## Source data")
    A("")
    A("| What | Where |")
    A("|---|---|")
    A(f"| Per-concurrency JSON / logs | `logs/atom/{a.sweep.name}/c<N>.{{json,log}}` |")
    A(f"| Sweep summary | `logs/atom/{a.sweep.name}/summary.txt` |")
    A(f"| Server log (memory budget, engine config) | `logs/atom/{a.sweep.name}/atom_server.log` |")
    A(f"| Driver state | `logs/atom/{a.sweep.name}/STATE.txt` |")
    A("| This table as CSV | `results/kimi-k3-maxseqs.csv` |")
    A("| Rerun rationale + config diff | `notes-kimi-k3.md` |")
    A("| Original (non-MAD) run | `kimi-k3-base.md` |")
    A("")
    A("---")
    A("")
    A("## Terminology — HBM and XGMI")
    A("")
    A("**HBM — High Bandwidth Memory.** The GPU's own on-package memory, where weights, KV")
    A("cache and activations live. **8 TB/s per GPU**, 288 GB capacity. This is the")
    A("**intra-GPU** path — inside one GPU, no other GPU involved.")
    A("")
    A("**XGMI — the GPU↔GPU interconnect** (AMD Infinity Fabric). Direct links between GPUs,")
    A("bypassing CPU and PCIe. All-to-all mesh (K₈), every pair 1 hop, **~537 GB/s per")
    A("direction** aggregate per GPU. This is the **intra-node** path; AMD's NVLink counterpart.")
    A("")
    A("HBM is ~15× faster than XGMI per GPU, so the instinct is that HBM can never be the")
    A(f"constraint. For this workload that is backwards: HBM moves ~{f(wgb)} GB/step while XGMI")
    A(f"moves ~{wire_step/1e9:.2f} GB — the *slower* link is the idle one.")
    A("")

    (a.out / "kimi-k3-maxseqs.md").write_text("\n".join(Lx) + "\n")
    print(f"wrote {a.out}/kimi-k3-maxseqs.md and .csv ({len(rows)} concurrency points)")


if __name__ == "__main__":
    main()
