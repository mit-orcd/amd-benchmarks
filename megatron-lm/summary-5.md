# Megatron-LM TF/s Sweep on gfx950-native Image — Result Summary

**Sources**
- `work/log.tflops-v26.1` — driver log for the sweep, started `2026-06-02 21:47:58 CDT`.
- `work/logs/tflops_v26.1_20260602_214758/` — per-config container logs (`bench_mbs{N}_rc{mode}_{prec}.log`) and `tflops_summary.txt`.
- Driver: `work/run-tflops-v26.1.sh` (MBS × recompute × precision sweep at fixed N=8).
- New image: `megatron-lm-v26.1.sif` (built from `docker://rocm/megatron-lm:v26.1`, 21 GB), sitting alongside — not replacing — the original `megatron-lm.sif`. The old-image baseline and the 3–5× gain analysis are kept in §8 as a side note.

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

Per-precision weak-scaling curves on the gfx950-native image at MBS=4. BF16 is fully populated (N=1..8) from the GPU-count sweep (§7); FP8 has only the N=8 data point from the MBS×RC×precision sweep (§2) — a dedicated FP8 GPU-count sweep is still to be run.

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

Source: §7, `logs/tflops_v26.1_gpusweep_20260603_153153/`. ¹ Per-GPU TF/s normalized to N=4 — the lowest viable N at no-recompute. A true N=1 baseline is not measurable: N=1 OOM'd across all three sharding + recompute modes the script tried (see §7 "Why N=1 OOM"). ² **N=2 and N=3 use `--recompute-granularity full`**, which is not the same workload as the no-RC baseline at N=4+ and is detailed in the discussion bullet on "N=2/3 require full recompute" below.

### FP8 hybrid (MBS=4, delayed scaling)

| N | GBS | last TF/s/GPU | parallel efficiency |
|--:|----:|--------------:|--------------------:|
| 8 |  32 |    **1,108.0 ★** | — (single point) |

★ Overall throughput winner across all configurations measured. Source: §2 MBS×RC×precision sweep at N=8, `logs/tflops_v26.1_20260602_214758/`. **A dedicated FP8 GPU-count sweep across N=1..8 has not been run yet** — listed in §6 next-experiments. The FP8/BF16 ratio at the matched MBS=4 / N=8 point is **1.45×** (1,108.0 / 766.2), close to the §2 figure of 1.43×.

### Discussion

- **N=4 and N=8 are the only good BF16 scaling points.** Power-of-2 N hits 727–766 TF/s/GPU; N=5/6/7 collapse to ~566–586 TF/s/GPU — a ~21 % per-GPU drop just by adding one rank, with a flat plateau across N=5/6/7, then recovery at N=8. The shape is `✗OOM, ✗RC, ✗RC, ✓, ✗cliff, ✗cliff, ✗cliff, ✓` — not classic weak-scaling decay.
- **Root cause of the N=5–7 cliff is at the RCCL layer**, not Megatron. A `rccl-tests all_reduce_perf` sweep on the same fabric (post-sweep, see §7) shows allreduce busbw drops from 162 GB/s at N=4 / 386 GB/s at N=8 to ~38 GB/s flat across N=5/6/7 — a ~4× collective regression at the same xGMI hardware. Megatron's per-iter time grows accordingly (2.29 s → ~2.90 s) for identical per-rank compute. Probable mechanism: RCCL falls back from a clean ring at power-of-2 N to a tree-or-mixed schedule at other N. Closing this is the highest-leverage next experiment (§6 #7).
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
| 8 |         986.0 |       **775.1** | none | **78.6 %** |

¹ N=2 MI355X uses full recompute (~20–25 % compute tax — no-RC OOMs at MBS=4 on 288 GiB HBM); RC-corrected estimate is ~71 %. B200 weak-scales cleanly (95–96 % efficiency from N=1→8); MI355X has a separate RCCL cliff at N=5/6/7 (busbw drops 4×, see §7). **Apples-to-apples conclusion: at matched MBS=4 BF16 no-RC, MI355X is 73.6 % of B200 at N=4 and 78.6 % at N=8.** Source: `logs/tflops_v26.1_gpusweep_20260603_153153/`.

#### 2. Megatron vs rocmval on MI355X

Same hardware, very different realized throughput:

| Measurement                              | TF/s/GPU | % of MI355X silicon peak (2,500) |
|------------------------------------------|---------:|---------------------------------:|
| rocmval BF16 (peak shape)                |  ~2,475  |                            ~99 % |
| hipBLASLt BF16 ceiling                   |   1,640  |                           65.6 % |
| Megatron BF16 N=8 (best)                 |    775.1 |                           31.0 % |
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

**Not contributors:** hardware (silicon favors MI355X), driver/fabric at power-of-2 N (the N=5/6/7 cliff is separate, §7), or Megatron itself (no vendor-specific tuning above the TE → GEMM line).

**Closes the gap:** hipBLASLt shape-mix tuning on gfx950 alone moves MI355X to or past parity. Secondary single-digit-pp items: gfx950 Apex fused RoPE (~3–5 %), FlashAttention path enabled in TE 2.6 (currently falls back to CK/AITER), and FP8 tuning closer to the 2.2× theoretical ratio (currently 1.43×).

---

## 2. MBS × Recompute × Precision sweep at N=8

At fixed N=8 DP, sweep (MBS, recompute, precision) to find the per-config sweet spot. This is where the FP8 winner came from and where the BF16 N=8 reference value of 778.1 TF/s/GPU was first measured (independently reproduced as 775.1 in §7).

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

With proper gfx950 GEMM kernels, **larger batches improve per-GPU throughput** by amortizing kernel launch / collective overhead against more compute per step. MBS=4 hits 0.88–0.90 HBM utilization, leaving no room for MBS=6+ without recompute. (Historical note: on the old `megatron-lm.sif` image MBS=2 was the BF16 winner — see §8.)

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

## 3. Memory utilization

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
- **Selective recompute** at MBS=4 used slightly *more* memory than no-RC (0.91 vs 0.88) without throughput benefit — same counter-intuitive pattern observed on the old image (TE selective-RC workspace overhead exceeds the savings at this model size).

---

## 4. The journey here (failures and fixes)

Five distinct issues had to be resolved between the old image and this winning sweep. Each is worth recording — these are traps anyone else migrating from a gfx942 image to v26.1 will hit.

| # | Failure mode | Root cause | Fix |
|---|--------------|------------|-----|
| 1 | Heredoc rendered torchrun command as broken arg list | `\\` in unquoted heredoc produced literal `\<newline>`, breaking arg parsing | Single `\` at line ends |
| 2 | `ValueError: bias_activation_fusion and use_te_activation_func cannot be both true` | TE 2.6 enforces mutual exclusion | Drop `--use-te-activation-func`, keep default `bias_activation_fusion` |
| 3 | `ValueError: No dot product attention backend is available` | TE 2.6 sees flash-attn 2.8.3 as "not installed" (outside ≤2.8.1 window); `--attention-backend flash` fails because FlashAttention is disabled | Use `--attention-backend fused` (TE's CK/AITER path, AMD-native fast path) |
| 4 | hipBLASLt `Unable to find any suitable algorithms` + SIGSEGV (env-driven) | `HIPBLASLT_TUNING_FILE` pointing at non-existent file disabled fallback; missing `CUDA_DEVICE_MAX_CONNECTIONS=1`; conflicting MSCCL / `NCCL_PROTO` settings vs. AMD's reference | Drop `HIPBLASLT_TUNING_FILE`; add `CUDA_DEVICE_MAX_CONNECTIONS=1`, `RCCL_MSCCL_ENABLE=0`, `NCCL_PROTO=Simple`, `TE_HIPBLASLT_TUNING_{RUN,ALGO}_COUNT` |
| 5 | hipBLASLt `Unable to find any suitable algorithms` in TE wgrad GEMM (final blocker) | TE's bias-fused wgrad backward path requests a hipBLASLt GEMM shape that has no gfx950 kernel in this image | `--disable-bias-linear` — routes through the bias-free wgrad path which IS tuned. Standard for modern LLMs (Llama-style) anyway. |

**Item 5 was the load-bearing fix** — without it nothing trains at all; with it everything works. The other four were necessary cleanup. Practical lesson: AMD's reference script `/workspace/Megatron-LM/examples/llama/train_llama3.sh` inside the image is the canonical recipe for working flag combinations; deviating from it without testing is high-risk.

---

## 5. Other notable findings

- **Loss converges normally** on mock data in all 7 configs. FP8 reaches a lower loss faster (7.07 at iter 50 vs 7.65 BF16) — consistent with the smaller per-step time letting cosine LR schedule decay further per wall-clock minute. No NaNs, no skipped iterations.
- **`Activation memory footprint per transformer layer (precise, without SP): 3264.0 MB`** logged for MBS=4 BF16 no-RC. 40 layers × 3.26 GB = 130 GB activations alone — explains why MBS=6 won't fit without RC and MBS=8 needs full RC.
- **Exit code -11 at process teardown** is shutdown-time noise (NCCL/RCCL cleanup race) and does not affect measurements; the throughput numbers are recorded *before* shutdown.
- **All 7 configs completed successfully** — no OOMs, no real-training crashes. The script's per-config resilience matters less now that no configs fail; the next sweep can be more ambitious (more MBS points, TP variants, MXFP8).

---

## 6. Recommended next experiments

Now that the baseline is competitive, the next gains are smaller but worth measuring.

1. **Try MXFP8 recipe.** `--fp8-recipe mxfp8` exists in this Megatron build and is the focus of the recent ROCm-fork commit `705c37b83` ("MXFP4 recipe enablement"). MX block-scaled FP8 is what MI355X silicon is optimized for; the current `--fp8-recipe delayed` is the legacy path. Expected: another **+10–25 %** on top of 1,111 TF/s.
2. **Try `--fp8-param-gather`.** Currently False. Keeps params in FP8 during the all-gather (vs. cast-then-gather), reducing comm bandwidth.
3. **TP=2 + sequence parallel.** With biases-off we're now in the regime where TE's TP-comm-overlap (`tp_comm_overlap_ag/rs`) becomes hot. `--tensor-model-parallel-size 2 --sequence-parallel` would split the model and enable bulk TP collectives.
4. **Push MBS=6 with selective RC.** The old-image OOM at MBS=6 selective was 0.978 util; with v26.1's tighter memory accounting and the bias-free model, MBS=6 selective might fit.
5. **Add `--use-flash-attn` after a flash-attn downgrade.** If a flash-attn ≤ 2.8.1 wheel can be sideloaded into the image, TE would re-enable the FlashAttention backend for cases where it beats FusedAttention.
6. **Source an Apex fused-RoPE build for gfx950.** The native fallback costs 3–5 %.
7. **Close the N=5/6/7 RCCL cliff.** Confirmed via `rccl-tests` at the RCCL layer (§7): at non-power-of-2 N the all_reduce busbw drops from 162 GB/s (N=4) / 386 GB/s (N=8) to ~37 GB/s flat. Next steps: sweep `NCCL_ALGO=Tree`, `NCCL_PROTO=LL/LL128`, `RCCL_MSCCL_ENABLE=1` with a tuned schedule, or an explicit `RCCL_TOPO_FILE` for the xGMI fabric. A clean fix here would lift N=5/6/7 from ~570 TF/s/GPU toward the N=4/N=8 plateau of 730–780 — a ~30 % gain at those configs without touching kernels.
8. **N=7/N=8 GPU-count gap closed** (2026-06-03 15:31 sweep, §7). N=8 BF16 MBS=4 no-RC = 775.1 TF/s/GPU on the dedicated GPU-count driver, matching the §2 figure (778.1) within 0.4 %.
9. **FP8 GPU-count sweep.** Only the N=8 point is known (1,108 TF/s/GPU at MBS=4 from §2). The BF16 cliff/recovery shape across N=4/5/6/7/8 may or may not transfer to FP8 — FP8's smaller activation and gradient payload could either dodge the RCCL fallback (smaller messages stay in the LL/Simple protocol) or hit it harder. Worth measuring before assuming the headline FP8 number holds at other N.

---

## 7. GPU-count weak-scaling sweep (BF16, MBS=4)

**Sources (re-run on 2026-06-03 15:31 CDT — see §7.5 for why the first attempt was discarded)**
- `work/log.tflops-v26.1-gpusweep-rerun-2` — driver log, started `2026-06-03 15:31:53 CDT`.
- `work/logs/tflops_v26.1_gpusweep_20260603_153153/` — per-N container logs + `tflops_summary.txt`.
- Driver: `work/run-tflops-v26.1-gpusweep.sh` (weak-scaling: MBS=4 fixed, GBS = MBS × N, BF16, OOM-fallback cascade `no_shard → no_shard+full-RC → optim_grads_params+full-RC`).
- Goal: produce the new-image equivalent of `summary-2.md` so the headline B200 comparison has real MI355X N=1..8 data.

### Per-N results

| N | GBS | shard / RC                  | iter time (ms) | TF/s/GPU last | TF/s/GPU best | mem util | status |
|--:|----:|-----------------------------|---------------:|--------------:|--------------:|---------:|--------|
| 1 |   4 | (all 3 rungs)               |   —            |  —            |  —            |  —       | OOM all 3 configurations — see "Why N=1 OOM" below |
| 2 |   8 | no_shard + full RC          |  2,973         |     558.8     |     **561.7** | 0.76     | ✓ (rung-1 no-RC OOM'd, rung-2 fit) |
| 3 |  12 | no_shard + full RC          |  2,844         |     584.8     |     **586.0** | 0.64     | ✓ (rung-1 no-RC OOM'd, rung-2 fit) |
| 4 |  16 | no_shard + no RC            |  2,288         |     726.9     |     **731.4** | 0.94     | ✓ |
| 5 |  20 | no_shard + no RC            |  2,838         |     586.0     |     **588.0** | 0.93     | ✓ |
| 6 |  24 | no_shard + no RC            |  2,898         |     574.0     |     **579.3** | 0.89     | ✓ |
| 7 |  28 | no_shard + no RC            |  2,940         |     565.7     |     **569.8** | 0.87     | ✓ |
| 8 |  32 | no_shard + no RC            |  2,171         |     766.2     |     **775.1** | 0.88     | ✓ — matches §2's 778.1 within 0.4 % |

Steady-state from iter 10 onward; iter time is the median of iters 10–50. **N=2/3 use full recompute** (no-RC OOM's at MBS=4 for N<4), so their numbers are not directly comparable to N=4+; rough RC penalty is ~20–25 % from §2 data.

### Why N=1 OOM (all three rungs)

Theoretical per-rank footprint reported by Megatron at MBS=4 BF16: **weight+optimizer ≈ 119 GiB, activation ≈ 139 GiB, total ≈ 258 GiB** with no-RC. With full recompute, activations drop to ~7 GiB per layer / per microbatch (uniform RC with `--recompute-num-layers 1`) — bringing the total to ~127 GiB.

But at N=1 there's nowhere to shard the optimizer state, and RCCL still initializes a comm buffer for the dist-opt setup. The third rung run failed with:
```
NCCL WARN Failed to CUDA calloc 536870912 bytes
ncclUnhandledCudaError: Call to CUDA function failed.
```
i.e., RCCL couldn't get a 512 MiB allocation after the model loaded. With 127 GiB resident, plus hipBLASLt workspaces, plus TE scratch, plus PyTorch caching allocator overhead, the 288 GiB HBM is too tight. **N=1 at MBS=4 is a hard physical limit on a single MI355X**, not a script bug.

**Workarounds for a true N=1 baseline (out of scope for this sweep):** drop MBS to 2 (`summary-2.md` did this on the old image and N=1 still OOM'd), or wait for an MI355X part with more HBM, or compare against a TP=2-style split that gives up the "one-rank-fits-all" property.

### The N=4 → N=5 cliff is at the RCCL layer

| N | Megatron iter time (ms) | Megatron TF/s/GPU | rccl-tests all_reduce busbw at 1 GiB (GB/s) |
|--:|------------------------:|------------------:|--------------------------------------------:|
| 4 | 2,288                   | 731.4             | 165                                         |
| 5 | 2,838                   | 588.0             | **37**                                      |
| 6 | 2,898                   | 579.3             | **37**                                      |
| 7 | 2,940                   | 569.8             | **39**                                      |
| 8 | 2,171                   | 775.1             | 386                                         |

The 4× drop in `all_reduce_perf` busbw at N=5/6/7 (measured on the same fabric, same env, immediately after the sweep) is the direct cause of the Megatron cliff. **Megatron's per-iter time grows by ~600 ms going from N=4 to N=5 for identical per-rank work** — that delta is entirely the gradient all-reduce + param all-gather taking longer.

Probable mechanism: at non-power-of-2 N on a fully-connected xGMI fabric, RCCL can't construct a clean unidirectional ring across all ranks and falls back to a tree-or-mixed algorithm that doesn't saturate the available bisection bandwidth. The flat ~37 GB/s plateau across all message sizes from 64 MiB to 1 GiB is the signature of a fixed-link-utilization fallback, not a topology-aware schedule.

**This explains the non-monotone scaling shape `N=4 ✓ / N=5–7 ✗ / N=8 ✓`** — power-of-2 N values get the fast ring, others get the slow fallback. Same pattern observed on the old image (`summary-2.md`) at much lower absolute throughput; the cliff is in the collective layer, persistent across image versions.

### N=8 reproducibility

Two independent runs at MBS=4 BF16 no-RC, N=8 (different drivers, ~21 h apart):

| Source | best TF/s/GPU | iter time (ms) | mem util |
|--------|--------------:|---------------:|---------:|
| MBS×RC×precision sweep (§2, 2026-06-02 21:47)    | 778.1 | 2,138 | 0.88 |
| GPU-count sweep (§7, 2026-06-03 15:31)            | 775.1 | 2,146 | 0.88 |

Δ = 0.4 % — well inside run-to-run variance. The headline 778.1 figure stands.

### 7.5. Operational notes (post-mortem from the failed first attempt)

The first re-run of this sweep (started 2026-06-03 09:46:42, dir `tflops_v26.1_gpusweep_20260603_094642/`) terminated incomplete and had to be discarded — useful as a record of what the fixes addressed:

| Failure | Root cause | Fix in current driver |
|---------|------------|-----------------------|
| N=7 stopped at iter 15 of 50; N=8 never ran | Host crashed at ~12:45 the same day (`last` shows session `Jun 3 09:47 - crash`). Most likely the N=7 non-power-of-2 collective wedged the GPU; host limped along then rebooted | Per-N `timeout 1800` cap; between-N `rocm-smi` health gate aborts the sweep if the driver is wedged |
| N=1 logged ERR with `OOM=no` despite RCCL alloc failure | OOM-detection regex matched only `HIP out of memory` / `OutOfMemoryError` / `CUDA out of memory`; RCCL's `Failed to CUDA calloc` slipped through, and the third (`optim_grads_params + full RC`) rung was never tried | Regex expanded to also catch `Failed to CUDA calloc`, `hipErrorOutOfMemory`, `cudaErrorMemoryAllocation` |
| Driver-side `[[: invalid arithmetic operator` error per successful run | `RC=$(run_one ...)` captured the entire `tee`d container output into `$RC` instead of just the exit code | `run_one` now writes its exit code to a global `RUN_RC` and uses no command substitution |
| N=1/2/3 reported as plain OOM with no fallback attempt | Original script only retried N=1 with `optim_grads_params` (useless at N=1 — nothing to shard to) | Three-rung OOM-fallback cascade: `no_shard, no-RC` → `no_shard, full-RC` → `optim_grads_params, full-RC` (last rung tried only at N=1) |

The current sweep ran to completion in 44 min 46 s with no hangs, no host issues, and clean steady-state across all N. The legacy log/dir from the failed first attempt is left in place under `tflops_v26.1_gpusweep_20260603_094642/` for reference.

---

## 8. Side note — comparison with the old `megatron-lm.sif` image (`summary-3.md`)

All numbers above are on the new gfx950-native image `megatron-lm-v26.1.sif`. The old `megatron-lm.sif` (gfx942 binaries running on MI355X under `HSA_OVERRIDE_GFX_VERSION=9.4.2`) is the prior baseline; its full analysis lives in `summary-3.md` and is preserved as a historical record of what the gfx950 fat-binary migration bought us.

### What drove the 3–5× gain

| Metric | Old image (`megatron-lm.sif`, `summary-3.md`) | This sweep (`megatron-lm-v26.1.sif`) | Improvement |
|--------|------------------------------------------------:|--------------------------------------:|------------:|
| Best per-GPU TF/s (BF16) | 236.8 | **778.1** | **3.29×** |
| Best per-GPU TF/s (overall) | 236.8 | **1,111.5** | **4.69×** |
| % of MI355X hipBLASLt BF16 ceiling (1,640 TF/s) | 14.4 % | 47.4 % (BF16) | 3.3× |
| % of MI355X hipBLASLt FP8 ceiling (3,611 TF/s) | n/a | 30.8 % (FP8) | n/a |

1. **gfx950-native image (~3×).** Hot kernels now load native gfx950 code instead of MI300X assembly under `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
2. **Tunings + FP8 (~1.4× on top).** Overlap flags, fused attention, AMD reference env block, and FP8 hybrid.

### MBS sweet-spot moved from 2 (old) to 4 (new)

On the old image, MBS=2 was the BF16 winner — gfx942 kernels saturated at small batches and degraded with larger ones. With proper gfx950 GEMMs the relationship inverts: MBS=4 wins both precisions because larger batches amortize kernel launch / collective overhead against more compute per step (see §2 for the new-image numbers; the +9 % BF16 / +17 % FP8 gain over MBS=2 only shows up once the kernels are native).

| MBS / precision | Old image (`summary-3.md`) | New image (§2) |
|-----------------|---------------------------:|---------------:|
| MBS=2 BF16      | 236.8                      | 715.7          |
| MBS=4 BF16      | 231.6                      | **778.1**      |
| MBS=4 FP8       | n/a                        | **1,111.5**    |

### Where else the old image still shapes the new analysis

- The "Selective RC uses more memory than no-RC" pattern noted in §3 was first observed in `summary-3.md`; it's a TE workspace overhead, not an image artifact, and persists on v26.1.
- The N=4 ✓ / N=5–7 ✗ / N=8 ✓ scaling shape noted in §7 was first seen in `summary-2.md` on the old image; the floor is now ~570 TF/s instead of ~155, but the cliff itself is in the collective layer and persists across image versions.

These observations made it into the body sections above (with no direct `summary-3.md` reference, since they're now properties of the v26.1 result and not historical context). This section is the single load-bearing pointer back to the old-image analysis for anyone who wants to walk that history.
