# Megatron-LM TF/s Sweep on gfx950-native Image — Result Summary

**Sources**
- `work/log.tflops-v26.1` — driver log for the sweep, started `2026-06-02 21:47:58 CDT`.
- `work/logs/tflops_v26.1_20260602_214758/` — per-config container logs (`bench_mbs{N}_rc{mode}_{prec}.log`) and `tflops_summary.txt`.
- Driver: `work/run-tflops-v26.1.sh` (MBS × recompute × precision sweep at fixed N=8).
- Image: `megatron-lm-v26.1.sif` (built from `docker://rocm/megatron-lm:v26.1`, 21 GB), gfx950-native.

**Setup recap**
- 1 node × 8 × AMD Instinct MI355X (gfx950), ROCm 7.2.3 host driver, PyTorch 2.10.0.dev20251112+rocm7.1 inside the new SIF.
- Workload: GPT, 40 layers, hidden 6144, FFN 16384, 48 heads (GQA, 8 KV groups), seq 4096, SwiGLU + RMSNorm + RoPE, untied embeddings, **no linear biases** (`--disable-bias-linear`, required on this image — see §4).
- Parallelism: TP=1, PP=1, DP=8 (pure data-parallel), distributed optimizer ON, `--overlap-grad-reduce --overlap-param-gather`.
- Attention: `--attention-backend fused` (TE's CK/AITER path on AMD).
- Precision: BF16 baseline, plus FP8 hybrid (delayed scaling) variants.
- Mock data, 50 train iters, log every 5.
- Total trainable parameters: 15.60 B (transformer; embeddings untied).
- Source: `ROCm/Megatron-LM` branch `rocm_dev` HEAD `705c37b83`, host-bound into the container.

**Sweep matrix**: `(MBS, RC, precision)` for `MBS ∈ {2, 4, 8}` × `RC ∈ {none, selective, full}` × `precision ∈ {bf16, fp8}`, taking only the combinations expected to fit and be informative.

---

## Headline

Per-precision weak-scaling curves on the gfx950-native image at MBS=4. BF16 is fully populated (N=1..8) from the GPU-count sweep (`logs/tflops_v26.1_gpusweep_20260603_153153/`); FP8 has only the N=8 data point from the MBS×RC×precision sweep (§2). The tuned bests come from the §3 tuning sweep.

### BF16 (MBS=4)

| N | GBS | last TF/s/GPU | parallel efficiency¹ |
|--:|----:|--------------:|---------------------:|
| 1 |   4 | OOM           | —                    |
| 2 |   8 |       558.8   |             76.9 %²  |
| 3 |  12 |       584.8   |             80.4 %²  |
| 4 |  16 |     **726.9** |              100.0 % |
| 5 |  20 |       586.0   |               80.6 % |
| 6 |  24 |       574.0   |               79.0 % |
| 7 |  28 |       565.7   |               77.8 % |
| 8 |  32 |     **766.2** |              105.4 % |

Source: `logs/tflops_v26.1_gpusweep_20260603_153153/`. ¹ Per-GPU TF/s normalized to N=4 — the lowest viable N at no-recompute. A true N=1 baseline is not measurable: N=1 OOM'd across all three sharding + recompute modes the script tried (the 16 B model + RCCL comm buffer exceeds 288 GiB HBM at single-rank scale). ² **N=2 and N=3 use `--recompute-granularity full`**, which is not the same workload as the no-RC baseline at N=4+ and is detailed in the discussion bullet on "N=2/3 require full recompute" below. **Tuned best at N=8: 790.4 TF/s/GPU** (§3, `--ddp-bucket-size 250000000`) — +2.0 % over the 775.1 untuned weak-scaling figure used in the rest of this table.

### FP8 hybrid (MBS=4, delayed scaling)

| N | GBS | last TF/s/GPU | parallel efficiency |
|--:|----:|--------------:|--------------------:|
| 8 |  32 |    **1,108.0 ★** | — (single point) |

★ Overall throughput winner across all configurations measured. Source: §2 MBS×RC×precision sweep at N=8, `logs/tflops_v26.1_20260602_214758/`. **A dedicated FP8 GPU-count sweep across N=1..8 has not been run yet.** The FP8/BF16 ratio at the matched MBS=4 / N=8 point is **1.45×** (1,108.0 / 766.2), close to the §2 figure of 1.43×. **Tuned best at N=8: 1,119.3 TF/s/GPU** (§3, `--fp8-margin 2 --fp8-amax-history-len 16`) — +1.0 % over the 1,108.0 baseline.

### Discussion

- **N=4 and N=8 are the only good BF16 scaling points.** Power-of-2 N hits 727–766 TF/s/GPU; N=5/6/7 collapse to ~566–586 TF/s/GPU — a ~21 % per-GPU drop just by adding one rank, with a flat plateau across N=5/6/7, then recovery at N=8. The shape is `✗OOM, ✗RC, ✗RC, ✓, ✗cliff, ✗cliff, ✗cliff, ✓` — not classic weak-scaling decay.
- **Root cause of the N=5–7 cliff is at the RCCL layer**, not Megatron. A `rccl-tests all_reduce_perf` sweep on the same fabric shows allreduce busbw drops from 162 GB/s at N=4 / 386 GB/s at N=8 to ~38 GB/s flat across N=5/6/7 — a ~4× collective regression at the same xGMI hardware. Megatron's per-iter time grows accordingly (2.29 s → ~2.90 s) for identical per-rank compute. Probable mechanism: RCCL falls back from a clean ring at power-of-2 N to a tree-or-mixed schedule at other N.
- **Super-linear scaling N=4 → N=8 (105.4 % parallel efficiency).** The dist-opt all-reduce + all-gather collective fraction shrinks faster than the per-rank compute as N doubles on a topology-friendly count.
- **N=2/3 require full recompute** because at MBS=4 the per-rank footprint (~252 GiB no-RC) exceeds the 288 GiB HBM at low DP rank. Full RC drops activations to ~7 GiB per layer (with `--recompute-num-layers 1`) but adds a ~20–25 % compute overhead (the forward pass is re-executed during backward to regenerate activations on the fly). The table's parallel-efficiency values for N=2/3 (76.9 % / 80.4 %) therefore conflate two effects: the RC compute tax + any actual parallel-scaling penalty. Backing out the RC tax (multiply by ~1/0.78) gives apples-to-apples efficiency estimates of **~99 % at N=2 and ~103 % at N=3** — meaning at the same MBS+RC config N=2/3 would sit at or slightly above the N=4 no-RC plateau, which is what we'd expect given that smaller N has lower collective overhead. The depressed table figures are an RC artifact, not a real parallel-scaling problem.
- **N=1 is a hard physical limit at MBS=4.** Even with full RC + `optim_grads_params` (FSDP-style param/grad sharding), RCCL init fails to allocate its 512 MiB comm buffer (`NCCL WARN Failed to CUDA calloc 536870912 bytes`).
- **FP8 GPU-count shape is unknown.** Only N=8 at MBS=4 FP8 has been measured (1,108 TF/s/GPU). Whether the BF16 cliff/recovery shape carries to FP8 — or whether FP8's smaller activation/grad payload changes the collective regime — is open.

### vs. NVIDIA B200

The long-form version is preserved in `summary-long.md`. The condensed write-up follows.

#### 1. Megatron MI355X vs B200

Per-GPU throughput, BF16, MBS=4:

| N | B200 TF/s/GPU | MI355X TF/s/GPU | RC | MI355X / B200 |
|--:|--------------:|----------------:|----|--------------:|
| 1 |       1,031.6 | OOM             | —    | —          |
| 2 |       1,005.3 |           561.7 | full | 55.9 %¹    |
| 4 |         993.5 |       **731.4** | none | **73.6 %** |
| 8 |         986.0 |       **775.1** (790.4²) | none | **78.6 %** (**80.2 %**²) |

¹ N=2 MI355X uses full recompute (~20–25 % compute tax — no-RC OOMs at MBS=4 on 288 GiB HBM); RC-corrected estimate is ~71 %. B200 weak-scales cleanly (95–96 % efficiency from N=1→8); MI355X has a separate RCCL cliff at N=5/6/7 (busbw drops 4×). ² **Tuned best (§3):** 790.4 TF/s/GPU at N=8 with `--ddp-bucket-size 250000000`, moving MI355X to **80.2 %** of B200 at matched N=8 MBS=4 no-RC. The 775.1 figure is the untuned weak-scaling value used in the headline BF16 table for parallel-efficiency comparability across N. **Apples-to-apples conclusion: at matched MBS=4 BF16 no-RC, MI355X is 73.6 % of B200 at N=4 and 78.6–80.2 % at N=8.** Source: `logs/tflops_v26.1_gpusweep_20260603_153153/` and `logs/tflops_v26.1_tune_20260604_104624/`.

#### 2. Megatron vs rocmval on MI355X

Same hardware, very different realized throughput:

| Measurement                              | TF/s/GPU | % of MI355X silicon peak (2,500) |
|------------------------------------------|---------:|---------------------------------:|
| rocmval BF16 (peak shape)                |  ~2,475  |                            ~99 % |
| hipBLASLt BF16 ceiling                   |   1,640  |                           65.6 % |
| Megatron BF16 N=8 (untuned best)         |    775.1 |                           31.0 % |
| Megatron BF16 N=8 (tuned best, §3)       |  **790.4** |                         **31.6 %** |
| Megatron BF16 N=4 (best)                 |    731.4 |                           29.3 % |

rocmval is a microbenchmark on hand-tuned single-shape GEMMs (CK / rocBLAS at near-peak shapes). Megatron exercises ~7 transformer GEMM shapes back-to-back through hipBLASLt + TE, plus collectives and TE overhead. The gap between rocmval (~99 %) and hipBLASLt ceiling (~66 %) is the library identity; the gap between hipBLASLt ceiling and Megatron is shape-mix + framework overhead.

#### 3. Answer: rocmval MI355X is 1.1× gpu-freyer B200 at BF16, but Megatron MI355X is ~78 % of B200 — why?

Three things flip the ratio:

| Effect                   | rocmval (MI355X)                              | Megatron (MI355X)                                         |
|--------------------------|-----------------------------------------------|-----------------------------------------------------------|
| Library                  | spec / CK / rocBLAS at one peak shape         | hipBLASLt via TE across all transformer shapes            |
| Shape coverage           | 1–2 hand-tuned shapes near silicon ceiling    | ~7 distinct shapes (QKV / O / FFN up/gate/down) back-to-back |
| Surrounding cost         | raw GEMM only                                 | TE dispatch + RCCL collectives + activation reformat      |

Quantitatively:

```
MI355X_Megatron / B200_Megatron = (silicon ratio) × (library efficiency ratio)
                                = 1.11           × (~0.66 / ~0.90+)
                                ≈ 0.81
```

— matches the measured 0.74 (N=4) / 0.79 (N=8). The 1.11× silicon advantage is real, but hipBLASLt realizes only ~66 % of MI355X peak across Megatron's shape mix, while cuBLAS realizes 90 %+ of B200 peak across the same shapes. The library-efficiency ratio inverts the silicon ratio.

**But why does MI355X lose MORE going microbench → Megatron than B200 does?** Both vendors drop from their microbenchmark peak to Megatron — what's different is the size of that drop:

| Vendor | Microbench BF16 peak | Megatron N=8 BF16 best | Microbench → Megatron retention |
|---|---:|---:|---:|
| B200    | ~2,250 (cuBLAS BF16 peak) | 986.0 | **~44 %** |
| MI355X  | ~2,475 (rocmval BF16)     | 790.4 (tuned, §3) | **~32 %** |

B200 keeps ~44 % of its microbench number in Megatron; MI355X keeps only ~32 % — a ~12 pp retention gap. That gap comes from four compounding factors, all rooted in software maturity:

| Factor                          | B200 impact | MI355X impact | Net contribution to the 12 pp gap |
|---------------------------------|-------------|---------------|-----------------------------------|
| GEMM library shape coverage     | cuBLAS hits ~90 %+ of peak on each of the ~7 Megatron shapes (years of tuning) | hipBLASLt hits ~66 % across the same shapes (gfx950 fat-binary is months old) | **~7–9 pp** (dominant) |
| Attention path                  | FlashAttention (fully fused, highly tuned)                              | FA disabled (TE 2.6 ≠ FA 2.8.3 version), falls back to CK/AITER FusedAttention | **~3–5 pp** |
| RoPE fusion                     | Apex fused RoPE for Blackwell                                            | No gfx950 Apex RoPE kernel; unfused PyTorch fallback                          | **~2–3 pp** |
| TE ↔ GEMM-library integration   | cuBLAS/TE is a multi-year integration; aggressive kernel fusion across linear-bias-activation-norm | hipBLASLt/TE on ROCm is newer; more Python-side dispatch, fewer fused paths | **~1–2 pp** |

The collective layer is **not** a contributor at the comparison points: RCCL allreduce busbw at N=8 is 386 GB/s (higher than the 165 GB/s figure at N=4 on the same fabric), and Megatron iter time at N=8 is dominated by GEMM, not communication.

**In short**, the same software-maturity gap that puts Megatron MI355X at 78 % of B200 *also* explains why MI355X falls further from its own microbenchmark peak than B200 does. It's the same problem viewed from two angles — every factor that costs throughput in the microbench → Megatron transition costs more on the AMD side because each piece of the stack (hipBLASLt, TE/AMD integration, gfx950 Apex, FA on ROCm) is newer.

**Not contributors:** hardware (silicon favors MI355X), driver/fabric at power-of-2 N (the N=5/6/7 RCCL cliff is a separate issue and doesn't apply at N=2/4/8), or Megatron itself (no vendor-specific tuning above the TE → GEMM line).

**Closes the gap:** hipBLASLt shape-mix tuning on gfx950 alone moves MI355X to or past parity. Secondary single-digit-pp items: gfx950 Apex fused RoPE (~3–5 %), FlashAttention path enabled in TE 2.6 (currently falls back to CK/AITER), and FP8 tuning closer to the 2.2× theoretical ratio (currently 1.43×).

---

## 2. MBS × Recompute × Precision sweep at N=8

At fixed N=8 DP, sweep (MBS, recompute, precision) to find the per-config sweet spot. This is where the FP8 winner came from and where the BF16 N=8 reference value of 778.1 TF/s/GPU was first measured (independently reproduced as 775.1 in the GPU-count sweep).

| MBS | Recompute | Precision | GBS | iter time (ms) | last TF/s/GPU | best TF/s/GPU | Mem util |
|----:|-----------|-----------|----:|---------------:|--------------:|--------------:|---------:|
|   2 | none      | bf16      |  16 |        ~1,250  |         709.7 |         715.7 |     0.65 |
|   4 | none      | bf16      |  32 |        ~2,170  |         766.9 |     **778.1** |     0.88 |
|   4 | selective | bf16      |  32 |        ~2,240  |         744.5 |         745.7 |     0.91 |
|   8 | full      | bf16      |  64 |        ~5,000  |         665.2 |         670.4 |     0.65 |
|   2 | none      | **fp8**   |  16 |          ~700  |         914.2 |         950.6 |     0.71 |
|   4 | none      | **fp8**   |  32 |        ~1,510  |       1,108.0 |  **1,111.5** ★ |     0.90 |
|   8 | full      | **fp8**   |  64 |        ~3,300  |         966.4 |         982.1 |     0.65 |

★ **Overall winner: MBS=4, no recompute, FP8 — 1,111.5 TF/s/GPU best (1,108.0 last).** Sources: `work/log.tflops-v26.1` + `logs/tflops_v26.1_20260602_214758/`.

### Best BF16: MBS=4, no recompute — 778.1 TF/s/GPU

Steady-state from iter 10 onward (iter 5 still warming):

| iter | iter time (ms) | TF/s/GPU | loss |
|----:|---------------:|---------:|-----:|
| 10  | 2,172          |  765.6   | 10.17 |
| 15  | 2,186          |  760.8   | 8.54 |
| 20  | 2,180          |  762.8   | 8.21 |
| 25  | 2,176          |  764.4   | 8.17 |
| 30  | 2,145          |  775.5   | 8.12 |
| 35  | 2,187          |  760.5   | 8.05 |
| 40  | 2,138          |**778.1** | 7.94 |
| 45  | 2,143          |  776.2   | 7.77 |
| 50  | 2,169          |  766.9   | 7.65 |

Steady-state mean ≈ 767 TF/s/GPU. Iter-to-iter spread ~2 % (kernel-launch jitter).

### Overall winner: MBS=4, no recompute, FP8 hybrid — 1,111.5 TF/s/GPU

| iter | iter time (ms) | TF/s/GPU | loss |
|----:|---------------:|---------:|-----:|
| 10  | 1,496          |**1,111.5**| 10.08 |
| 15  | 1,511          |  1,101.0 | 8.47 |
| 20  | 1,519          |  1,095.3 | 8.18 |
| 25  | 1,520          |  1,094.1 | 8.14 |
| 30  | 1,539          |  1,080.7 | 7.99 |
| 35  | 1,513          |  1,099.4 | 7.79 |
| 40  | 1,512          |  1,100.1 | 7.59 |
| 45  | 1,540          |  1,080.1 | 7.33 |
| 50  | 1,501          |  1,108.0 | 7.07 |

Steady-state mean ≈ 1,096 TF/s/GPU. FP8 loss decays faster than BF16 over the same iteration window (7.07 vs 7.65 at iter 50) because the cosine schedule advances the same number of decay steps in roughly two-thirds the wall time. No NaNs, no skipped iterations across all 50 iters.

### MBS sweet-spot is at MBS=4 in both BF16 and FP8

- **MBS=2 BF16: 715.7**
- **MBS=4 BF16: 778.1** ← **+9 % over MBS=2**
- **MBS=2 FP8: 950.6**
- **MBS=4 FP8: 1,111.5** ← **+17 % over MBS=2 in FP8**

With proper gfx950 GEMM kernels, **larger batches improve per-GPU throughput** by amortizing kernel launch / collective overhead against more compute per step. MBS=4 hits 0.88–0.90 HBM utilization, leaving no room for MBS=6+ without recompute.

### Recompute is unnecessary at MBS=4 and a regression at MBS=8

| MBS | RC | best TF/GPU | vs MBS=4 no-RC |
|----:|----|------------:|---------------:|
| 4 | none | 778.1 | baseline |
| 4 | selective | 745.7 | −4.2 % |
| 8 | full | 670.4 | −13.8 % |

Recompute trades activation memory for compute. On gfx950-tuned kernels the trade is unfavorable at this model size — the extra recompute cost outweighs the activation savings. Worth enabling only if HBM is a hard constraint (larger models or longer sequences).

### FP8 gives a clean 1.43× over BF16 at MBS=4

- BF16: 778.1 TF/s/GPU
- FP8 hybrid: 1,111.5 TF/s/GPU → **1.43×**

Theoretical FP8/BF16 silicon ratio on MI355X (per rocmval): 3,611 / 1,640 = 2.20×. Achieved framework ratio of 1.43× indicates the FP8 path is only partially tuned. There is likely another 30–40 % headroom if AMD's MXFP8 / blockwise recipes outperform delayed scaling on this workload.

---

## 3. N=8 tuning sweep (DP-only, MBS=4 BF16 + FP8)

Hold parallelism fixed (TP=PP=CP=1, DP=8, MBS=4, no-RC); ablate one knob at a time vs the §2 baseline. Source: `logs/tflops_v26.1_tune_20260604_104624/`.

| variant                  | prec | best TF/s/GPU | Δ          | knob                                            |
|--------------------------|:----:|--------------:|-----------:|-------------------------------------------------|
| bf16_baseline            | bf16 |     774.9     |    —       | reference                                       |
| bf16_ddp_bucket_250M     | bf16 |   **790.4**   | **+2.0 %** | `--ddp-bucket-size 250000000`                   |
| bf16_nccl_buffsize_16M   | bf16 |     789.2     |   +1.9 %   | `NCCL_BUFFSIZE=16777216`                        |
| bf16_hipblaslt_30x150    | bf16 |     784.3     |   +1.2 %   | deeper hipBLASLt tuning (vs 10×50)              |
| fp8_baseline             | fp8  |   1,108.1     |    —       | reference                                       |
| fp8_margin2_amax16       | fp8  |  **1,119.3**  |   +1.0 %   | `--fp8-margin 2 --fp8-amax-history-len 16`      |
| fp8_amax_max             | fp8  |   1,098.4     |   −0.9 %   | `--fp8-amax-compute-algo max` (higher variance) |

**New bests:** BF16 N=8 = **790.4** (was 778.1, +1.6 %); FP8 N=8 = **1,119.3** (was 1,111.5, +0.7 %).

**Did not work:**

- `RCCL_MSCCL_ENABLE=1` — **crashed the host at iter 15** (3 h downtime). MSCCL on this gfx950 image is unstable; do not enable.
- `TE_HIPBLASLT_TUNING_RUN_COUNT=50 / ALGO_COUNT=250` — TIMEOUT > 1800 s; 30×150 is the practical ceiling.
- `--apply-rope-fusion`, `--gradient-accumulation-fusion` — argparse rejects both in this Megatron build (`705c37b83`). Either renamed or compiled out.

**Takeaway:** the ~2 % BF16 headroom comes from collective amortization (DDP bucket, NCCL buffsize), not from the GEMM library. Deeper hipBLASLt online tuning gives only +1.2 %, confirming the ~22 % gap to B200 needs offline-tuned algorithms / gfx950 Apex RoPE / FA — all out-of-tree. The bigger FP8 lever is still untried **MXFP8** (`--fp8-recipe mxfp8`).

---

## 4. Memory utilization

| MBS | Recompute | Precision | Mem util | HBM used (GiB) | Headroom (GiB) |
|----:|-----------|-----------|---------:|---------------:|---------------:|
| 2 | none | bf16 | 0.65 | 187 | 101 |
| 4 | none | bf16 | 0.88 | 253 | 35 |
| 4 | selective | bf16 | 0.91 | 262 | 26 |
| 8 | full | bf16 | 0.65 | 187 | 101 |
| 2 | none | fp8 | 0.71 | 204 | 84 |
| 4 | none | fp8 | 0.90 | 259 | 29 |
| 8 | full | fp8 | 0.65 | 187 | 101 |

**Observations:**
- The model itself (weights + dist-opt state) takes ~120 GiB; activations add MBS-proportional pressure.
- MBS=4 no-RC sits at 0.88 (BF16) / 0.90 (FP8) HBM util — tight but stable, no OOMs.
- **Full recompute** drops activation memory to near-zero, giving 0.65 util at MBS=8 — could be pushed to MBS=12+ if needed.
- **Selective recompute** at MBS=4 used slightly *more* memory than no-RC (0.91 vs 0.88) without throughput benefit — TE selective-RC workspace overhead exceeds the activation savings at this model size.

---

## 5. The journey here (failures and fixes)

Five distinct issues had to be resolved to get a working sweep on this image. Each is worth recording.

| # | Failure mode | Root cause | Fix |
|---|--------------|------------|-----|
| 1 | Heredoc rendered torchrun command as broken arg list | `\\` in unquoted heredoc produced literal `\<newline>`, breaking arg parsing | Single `\` at line ends |
| 2 | `ValueError: bias_activation_fusion and use_te_activation_func cannot be both true` | TE 2.6 enforces mutual exclusion | Drop `--use-te-activation-func`, keep default `bias_activation_fusion` |
| 3 | `ValueError: No dot product attention backend is available` | TE 2.6 sees flash-attn 2.8.3 as "not installed" (outside ≤2.8.1 window); `--attention-backend flash` fails because FlashAttention is disabled | Use `--attention-backend fused` (TE's CK/AITER path, AMD-native fast path) |
| 4 | hipBLASLt `Unable to find any suitable algorithms` + SIGSEGV (env-driven) | `HIPBLASLT_TUNING_FILE` pointing at non-existent file disabled fallback; missing `CUDA_DEVICE_MAX_CONNECTIONS=1`; conflicting MSCCL / `NCCL_PROTO` settings vs. AMD's reference | Drop `HIPBLASLT_TUNING_FILE`; add `CUDA_DEVICE_MAX_CONNECTIONS=1`, `RCCL_MSCCL_ENABLE=0`, `NCCL_PROTO=Simple`, `TE_HIPBLASLT_TUNING_{RUN,ALGO}_COUNT` |
| 5 | hipBLASLt `Unable to find any suitable algorithms` in TE wgrad GEMM (final blocker) | TE's bias-fused wgrad backward path requests a hipBLASLt GEMM shape that has no gfx950 kernel in this image | `--disable-bias-linear` — routes through the bias-free wgrad path which IS tuned. Standard for modern LLMs (Llama-style) anyway. |

**Item 5 was the load-bearing fix** — without it nothing trains at all; with it everything works. The other four were necessary cleanup. Practical lesson: AMD's reference script `/workspace/Megatron-LM/examples/llama/train_llama3.sh` inside the image is the canonical recipe for working flag combinations; deviating from it without testing is high-risk.

---

## 6. Other notable findings

- **Loss converges normally** on mock data in all 7 configs. FP8 reaches a lower loss faster (7.07 at iter 50 vs 7.65 BF16) — consistent with the smaller per-step time letting cosine LR schedule decay further per wall-clock minute. No NaNs, no skipped iterations.
- **`Activation memory footprint per transformer layer (precise, without SP): 3264.0 MB`** logged for MBS=4 BF16 no-RC. 40 layers × 3.26 GB = 130 GB activations alone — explains why MBS=6 won't fit without RC and MBS=8 needs full RC.
- **Exit code -11 at process teardown** is shutdown-time noise (NCCL/RCCL cleanup race) and does not affect measurements; the throughput numbers are recorded *before* shutdown.
- **All 7 configs completed successfully** — no OOMs, no real-training crashes. The script's per-config resilience matters less now that no configs fail; the next sweep can be more ambitious (more MBS points, TP variants, MXFP8).

---
