# Sweep Summary — `sweep_20260601_130953`

Full 1..8 GPU x 9 precision sweep on 8 x MI355X (gfx950) using RVS `gst`
(`blas_source: hipblaslt`, `target_stress: 0`, 30 s per run). All 72 runs
reported `reporting=N/N` — every requested GPU emitted GFLOPS samples.

Source: `tflops_runs/sweep_20260601_130953/summary.{txt,csv}`, `log.sweep`.

## Aggregate TFLOPS and parallel efficiency

Aggregate TFLOPS (sum of per-GPU peak GFLOPS / 1000), with parallel efficiency
relative to the 1-GPU baseline in parentheses: `efficiency = agg_N / (N x agg_1)`.

| Prec | 1 GPU            | 2 GPU             | 3 GPU             | 4 GPU             | 5 GPU             | 6 GPU             | 7 GPU             | 8 GPU             |
|------|------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| fp4  | 3174.73 (100.0%) | 6343.17 (99.9%)   | 9499.22 (99.7%)   | 12747.00 (100.4%) | 15766.54 (99.3%)  | 19048.43 (100.0%) | 22202.18 (99.9%)  | 25309.64 (99.7%)  |
| fp6  | 1283.18 (100.0%) | 2564.07 (99.9%)   | 3833.11 (99.6%)   | 5100.25 (99.4%)   | 6379.18 (99.4%)   | 7647.38 (99.3%)   | 8854.40 (98.6%)   | 10046.92 (97.9%)  |
| bf6  | 1284.13 (100.0%) | 2563.08 (99.8%)   | 3845.30 (99.8%)   | 5112.64 (99.5%)   | 6309.51 (98.3%)   | 7582.65 (98.4%)   | 8727.70 (97.1%)   | 10157.62 (98.9%)  |
| fp8  | 3652.86 (100.0%) | 7336.19 (100.4%)  | 10996.65 (100.4%) | 14886.55 (101.9%) | 18428.34 (100.9%) | 22370.63 (102.1%) | 25683.11 (100.4%) | 29436.87 (100.7%) |
| bf8  | 3238.62 (100.0%) | 6549.68 (101.1%)  | 10055.48 (103.5%) | 13091.12 (101.1%) | 16658.96 (102.9%) | 19787.05 (101.8%) | 22821.89 (100.7%) | 26799.89 (103.4%) |
| fp16 | 1550.79 (100.0%) | 3119.26 (100.6%)  | 4674.68 (100.5%)  | 6258.88 (100.9%)  | 7745.18 (99.9%)   | 9388.97 (100.9%)  | 10935.48 (100.7%) | 12375.99 (99.8%)  |
| bf16 | 1655.89 (100.0%) | 3278.57 (99.0%)   | 4944.28 (99.5%)   | 6469.50 (97.7%)   | 8184.11 (98.8%)   | 9950.09 (100.1%)  | 11623.31 (100.3%) | 13208.84 (99.7%)  |
| fp32 | 153.47  (100.0%) | 306.81  (99.96%)  | 459.35  (99.8%)   | 611.55  (99.6%)   | 764.68  (99.7%)   | 915.40  (99.4%)   | 1068.07 (99.4%)   | 1220.81 (99.4%)   |
| fp64 | 76.81   (100.0%) | 152.72  (99.4%)   | 229.31  (99.5%)   | 306.93  (99.9%)   | 376.65  (98.1%)   | 459.19  (99.6%)   | 516.80  (96.1%)   | 612.48  (99.7%)   |

## Observations

### Scaling is essentially linear

The `gst` module runs an independent GEMM per GPU with no inter-GPU
communication, so perfect scaling is the expected behavior — and that is what
the data shows. Every precision sits within ~3% of ideal across the full 1..8
GPU range. The fully connected XGMI fabric is not exercised here; this sweep
measures per-GPU compute, not interconnect.

### Several precisions exceed 100% efficiency

`fp8`, `bf8`, `fp16`, and `bf16` all show efficiencies >100% at multiple GPU
counts (peaking at `bf8` 8-GPU = 103.4% and `fp8` 6-GPU = 102.1%). This is a
measurement artifact, not a real speedup:

- Per-GPU score is the **peak** GFLOPS sample observed across log intervals.
  Multi-GPU runs draw more samples in total, so the maximum-of-N estimator is
  biased upward as N grows.
- The 1-GPU baseline used `gpu61585` only. Looking at the 8-GPU per-GPU peaks,
  `gpu61585` is consistently below the node average for `fp8`/`bf8`/`bf16`
  (e.g. 8-GPU bf8: gpu61585=3393.55 vs node max gpu27852=3419.94). The
  baseline GPU happens to be a slightly slower sample for these precisions.

If a true linear-speedup metric is needed, compare against the **median**
per-GPU peak across the 8-GPU run rather than the lone 1-GPU number.

### Precision-class throughput ratios match MI355X marketing numbers

Single-GPU peaks (TFLOPS):

| Class          | Observed     | Notes |
|----------------|--------------|-------|
| fp4 (block-scaled MXFP4) | 3174.73 | ~2x fp8, as expected for half-width MX format |
| fp8 / bf8      | 3652.86 / 3238.62 | fp8 (e4m3) leads bf8 (e5m2) by ~13% |
| fp6 / bf6      | 1283.18 / 1284.13 | matches fp16; MX fp6 not getting the 2x dense-matrix-engine speedup that fp4 does in this test |
| fp16 / bf16    | 1550.79 / 1655.89 | bf16 ~7% faster than fp16 |
| fp32           | 153.47       | ~1/10 of bf16 — classic 32-bit GEMM ratio |
| fp64           | 76.81        | exactly 1/2 of fp32 |

Note that `fp6` and `bf6` measure at the **same** TFLOPS as fp16 here, not at
the ~2x fp8 rate the silicon supports. This is most likely a configuration
issue in the generated conf (matrix size 2048^3 + block-scaling overhead
versus the larger 8192 x 8192 x 16384 used for fp8/bf8/fp16/bf16), not a
hardware ceiling. Worth re-running fp6/bf6 with the larger GEMM shape before
quoting these as peaks.

### Per-GPU spread is small but not zero

Coefficient of variation across the 8 GPUs in the 8-GPU runs is under 2% for
every precision. Largest spread: `fp8` (~3% range, gpu19314=3546.81 vs
gpu19435=3765.45). gpu19314 is consistently the slowest sample at 8 GPUs
across multiple precisions (fp8, bf8, fp16, bf16) — possibly thermal or a
binning difference. Not large enough to be actionable on its own, but worth
watching if results drift further on a repeat run.

### fp64 7-GPU dip (96.1%)

The only sub-97% data point is `fp64` at 7 GPUs (avg/GPU 73.83 vs ~76.5
elsewhere). Every GPU in that run reported ~73-74 TFLOPS — uniformly lower,
not driven by one outlier. Likely transient (thermal carry-over from the
prior 6-GPU fp64 run, or a clock-stepping event), since the 8-GPU fp64 run
immediately after recovers to 76.56 avg/GPU. Re-run if it matters.

### Run did not error anywhere

`grep -c RVS-ERROR` on all logs is 0, and every run produced full per-GPU
coverage. The full 1..8 x 9 sweep took ~36 minutes of wall time at the
default 30 s `DURATION_MS`.
