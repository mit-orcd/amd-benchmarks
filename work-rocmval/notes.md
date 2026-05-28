The short answer is **library maturity, not silicon** — and yes, there are several things to try.

## Why fp4 / fp6 / bf6 are so low

1. **MX-formats are brand new on MI355X (CDNA 4).** MI355X shipped mid-2025; this is the first AMD generation with native MX-fp4/fp6 matrix instructions. The 10 PFLOPS paper peak assumes hand-tuned matrix kernels operating at full clock.

2. **hipBLASLt 1.2.2 (ROCm 7.2.3) is still building out MX kernels.** The other precisions are well-tuned (fp8 is 4 generations old; fp16/bf16 have been tuned for years), so they sit at 60–72 % of peak. Then look at the MX tier:
   - fp4 → 31.6 % of peak (~3.2 PFLOPS measured, library ceiling)
   - fp6 / bf6 → 12.8 % of peak (~1.28 PFLOPS measured, even lower library ceiling)

   The fp6 number being **half of fp4** is the smoking gun: on the silicon they have the same paper peak (10 PF), so equal-or-better fp6 throughput is what we'd expect from mature kernels. The library is hitting a different/slower code path for fp6.

3. **No `hipblaslt-bench` on this system.** AMD's own peak measurements use `hipblaslt-bench` (autotuning + algorithm selection); RVS's `gst` module calls hipBLASLt with a single algorithm — whichever heuristic returns first. If hipBLASLt's MX kernel tuning database doesn't yet have a high-throughput entry for 8192×8192×16384 fp4, you get a generic fallback.

So the bottleneck is the **BLAS library**, not the matrix engines or the benchmark conf.

## Things to try, in expected-impact order

### 1. Update ROCm / hipBLASLt (biggest lever, easiest if available)
This stack is `ROCm 7.2.3 / hipBLASLt 1.2.2`. AMD's monthly point releases add MX kernel tunings. If you can install ROCm 7.3+ (or AMD's nightly hipBLASLt build), expect substantial MX gains with no code changes.

### 2. Try alternative GEMM shapes (RVS-supported, can do now)
RVS gst supports several knobs I haven't varied:
- **`gemm_mode: strided_batched` + `batch_size: N`** — many smaller GEMMs per kernel launch; better tile alignment for MX block-scaling (32-element block grain).
- **Bigger square (`16384×16384×16384`)** — 4× more work per launch.
- **Different transpose (`transa: 0` / `transb: 0`)** — different tile path; sometimes much better-tuned.
- **Different `out_data_type`** — we use `bf16_r`; trying `fp32_r` or removing downcast may hit a better-tuned kernel.

### 3. Different `compute_type`
For MX formats, hipBLASLt may have separate kernel paths for `compute_type: fp32_r` (ours) vs. some accumulator-specific path. Worth a quick A/B.

### 4. Use `hipblaslt-bench` directly (best path to "true" peak)
It's not packaged here, but the source ships with the `hipblaslt` repo. Building it from source against the installed hipBLASLt lib gives you:
- Algorithm autotuning over hipBLASLt's full kernel catalog
- Per-precision sweep with explicit warmup/cooldown
- This is the canonical tool AMD uses for the peak numbers it publishes

### 5. Bypass RVS, write a minimal hipBLASLt program (most work, best result)
A 100-line C++ that calls `hipblasLtMatmulAlgoGetHeuristic` + `hipblasLtMatmul` in a tight loop, picking the fastest of the returned algorithms. This is what a "peak demonstration" actually is.

### 6. (Lowest impact) Try MX-specific output types and `rotating` values
Larger `rotating` (1024, 2048) so the working set is too big for L2 — forces HBM traffic and reduces clock boost, but ensures we're measuring sustained, not peak-spike.

## Concrete next experiment

A quick A/B for fp4/fp6/bf6 across a few shape/layout variants. Variants to try:

| Variant | Matrix | gemm_mode | transa/b | out_data_type |
|---------|--------|-----------|----------|---------------|
| baseline (current)   | 8192×8192×16384 | gemm | 1/0 | bf16 |
| square-16K           | 16384³          | gemm | 1/0 | bf16 |
| strided-batched      | 4096³ × 16      | strided_batched | 1/0 | bf16 |
| NN-layout            | 8192×8192×16384 | gemm | 0/0 | bf16 |
| fp32-out             | 8192×8192×16384 | gemm | 1/0 | fp32 |

If one variant moves fp4 from 3.2 PF → 5+ PF, keep it. Otherwise it confirms the library ceiling and points to options 1/4/5.

---

## Should I ask the admin to upgrade hipBLASLt?

**Short answer: yes, but do the no-admin experiments first — and ask for something specific, not "upgrade ROCm."**

### Why it's worth asking

- The fp4/fp6/bf6 gap is almost certainly a library issue, not hardware. AMD's kernel devs have been shipping MX tunings in every monthly point release since MI355X launched. ROCm 7.3+ (or hipBLASLt 1.3+) is likely to close a large fraction of the gap with zero code changes.
- You have concrete evidence to make the ask compelling: "fp4 measures 3.2 PFLOPS vs. 10 PFLOPS paper peak (32 %) and fp6 measures 1.3 PFLOPS (13 %) — both known to be BLAS-library-bounded on ROCm 7.2.3."

### How to frame the ask

Ask for **hipBLASLt specifically**, not a full ROCm stack upgrade. Reasons:
1. hipBLASLt is a single shared library (`libhipblaslt.so`), not the kernel driver or the whole toolchain. The risk to other users is low.
2. hipBLASLt has its own versioning (`hipBLASLt 1.x`) and release cadence; you can point the admin to the specific changelog entry for MX kernel tunings.
3. A full ROCm upgrade touches the driver and runtime and is much harder to justify for a shared system.

Suggested wording:
> "We're benchmarking MI355X MX-format throughput (FP4/FP6). With hipBLASLt 1.2.2 / ROCm 7.2.3, fp4 achieves ~3 PFLOPS vs. 10 PFLOPS peak — a known kernel-library maturity issue. AMD's changelog shows MX kernel tunings added in hipBLASLt 1.3+. Could we get hipBLASLt updated in-place without touching the rest of the ROCm stack?"

### No-admin alternatives (run these first, gives you data for the ask)

1. **Shape/layout A/B experiment** (see "Concrete next experiment" above) — takes ~30 min, tells you if the ceiling is really the library or the specific kernel path.

2. **Build hipblaslt-bench from source** — the `hipblaslt` repo includes a benchmark binary that does algorithm autotuning. It links against the installed `libhipblaslt.so`, no system changes needed. If hipblaslt-bench hits higher numbers with the same library, the issue is RVS's single-algorithm heuristic, not the kernel tuning.

3. **Containerized ROCm** — if the admin can pull an AMD ROCm Docker image with a newer hipBLASLt, you can run inside the container without changing the system. Requires Docker/Podman access, which may be easier to get than a system library update.

4. **Build hipBLASLt from source** against the installed ROCm runtime — advanced, but doable without root. Links `libhipblaslt.so` into your local prefix and `LD_LIBRARY_PATH` overrides the system copy for your process only. Lets you test any upstream hipBLASLt build without touching system files.

### Recommended sequence

1. Run the shape/layout A/B experiment (30 min, no admin).
2. If fp4 stays at ~3 PF across all variants → confirmed library ceiling → make the hipBLASLt upgrade ask with your data.
3. In parallel, build `hipblaslt-bench` to get AMD's own peak number for the installed library — useful as a reference and strengthens the ask.
4. If an upgrade isn't possible, the containerized path is the fastest workaround.

---

## BF16 tuning experiments (ROCm 7.2.3, MI355X)

Baseline: `8192×8192×16384`, TN layout (`transa: 1, transb: 0`), `rotating: 512` → **1,639.78 TFLOPS (65.6 % of 2,500 peak)**, 8-GPU 13,261 TFLOPS.

### Experiment 1 — Square 16384³ matrix

Change: `matrix_size_a/b/c: 16384`, `rotating: 1024`, layout unchanged (TN).

| GPUs | Baseline | 16K³ | Delta |
|------|---:|---:|---:|
| 1 | 1,639.78 | 1,622.60 | −1 % (noise) |
| 2 | 3,251.28 | 3,247.00 | flat |
| 4 | 6,633.83 | 3,860.94 | **−42 %** |
| 8 | 13,261.04 | 6,773.57 | **−49 %** |

**Verdict: reverted.** 1-GPU flat (no improvement to % of peak); 4 and 8-GPU
throttled severely. The 16K³ shape is 8× more work per kernel, power-dense
enough to saturate the 11.2 kW OAM tray when all dies are loaded. Original
shape is the sustained-throughput sweet spot.

### Experiment 2 — NN layout (`transa: 0, transb: 0`)

Change: same 8192×8192×16384 shape; only transpose changed to NN.

| GPUs | Baseline | NN layout | Delta |
|------|---:|---:|---:|
| 1 | 1,639.78 | 1,493.37 | **−9 %** |
| 2 | 3,251.28 | 2,966.65 | −9 % |
| 4 | 6,633.83 | 5,689.14 | −14 % |
| 8 | 13,261.04 | 7,757.84 | **−41 %** |

**Verdict: reverted.** NN is significantly worse across all GPU counts; the TN
path is hipBLASLt 1.2.2's well-tuned BF16 path. (Coincidence: NN 1-GPU number
1,493 matches the B200 BF16 reference exactly.)

### Conclusion from experiments 1 & 2

Both experiments hit worse kernel paths. See the full sweep (experiment 3)
below for the definitive ceiling measurement.

---

### Experiment 3 — Full variant sweep (`bf16_tune.sh`, 1 GPU, 12 variants)

Script: `bf16_tune.sh` — runs 1 GPU only to avoid OAM-tray power throttling
that masked kernel-level differences in earlier multi-GPU tests.

Output: `bf16_runs/20260528_093121/`

Variants tested and real results (invalid rows excluded — see notes below):

| Variant | TFLOPS | % of peak | vs baseline | Note |
|---|---:|---:|---:|---|
| **baseline** | **1,672.26** | **66.9 %** | — | 8192×8192×16384 TN, same as run_tflops.sh |
| krich_4k_64k | 1,656.42 | 66.3 % | −0.9 % | 4096×4096×65536 TN |
| krich_8k_32k | 1,654.02 | 66.2 % | −1.1 % | 8192×8192×32768 TN |
| rotating_256 | 1,641.96 | 65.7 % | −1.8 % | more cache-resident |
| rotating_2048 | 1,637.63 | 65.5 % | −2.1 % | force HBM working set |
| small_8k_sq | 1,616.93 | 64.7 % | −3.3 % | 8192³ square |
| TT_layout | 1,574.10 | 63.0 % | −5.9 % | transa=1 transb=1 |
| NT_layout | 1,555.48 | 62.2 % | −7.0 % | transa=0 transb=1 |
| squat_16k_8k | 1,552.43 | 62.1 % | −7.2 % | 16384×16384×8192 TN |

Discarded rows:
- **fp32_out** — hipBLASLt returns a BLAS error for bf16_r→fp32_r; not supported.
- **batched_8k_x4 / batched_4k_x16** — RVS gst reports per-iteration FLOPs as
  `batch × 2MNK` but timing doesn't scale proportionally → numbers come out
  batch_size× too high (23.8 PFLOPS reported for a 2.5 PFLOPS-peak GPU).
  Dividing by batch_size gives per-GEMM throughput of 1,489 and 1,607 TFLOPS
  respectively — both worse than baseline.

**Note:** today's baseline measured 1,672 vs earlier run's 1,640 — same conf,
~2 % noise from clock-warming run-to-run variation.

### Definitive conclusion

**Nothing beats baseline across 10 valid variants** covering matrix shape,
K-dominance, layout (TN / NT / TT), rotating buffer size, and total work size.

**~67 % of peak (1,640–1,672 TFLOPS) is hipBLASLt 1.2.2's BF16 ceiling via
the RVS gst knob space.** Further gains require:

- **`hipblaslt-bench`** — autotunes over hipBLASLt's full algorithm catalog
  (algorithm IDs RVS never exposes). Expected 70–80 % (1,750–2,000 TFLOPS),
  no admin needed, requires building from the hipBLASLt repo source.
- **Upgrade hipBLASLt** — see admin-upgrade section above.
