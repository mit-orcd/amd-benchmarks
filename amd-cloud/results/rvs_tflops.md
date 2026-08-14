# RVS `gst` TFLOPS — MI355X x8 (gfx950, ROCm 7.14)

System: 8 x AMD Instinct MI355X (CDNA 4 / gfx950), ROCm 7.14, Ubuntu 22.04.5.
Source runs: sweep_20260813_184243

## What this benchmark does

`run_tflops.sh` drives the ROCm Validation Suite (RVS) `gst` module (hipBLASLt GEMM
kernels) to measure sustained matrix-multiply throughput. For each precision and GPU
count, a YAML conf is generated from the shipped `conf/MI355X/levels/rvs_level_5.conf`
template, RVS runs with `parallel: true` so every selected GPU runs the GEMM
concurrently, and each GPU emits `GFLOPS <n>` every 3 s. The script takes the **peak**
per-GPU value (steady-state proxy) and sums across GPUs for the aggregate.
`target_stress: 0` means it measures only -- no pass/fail threshold.

Every GPU runs an **independent** GEMM: no XGMI or PCIe traffic, no RCCL. Scaling is
therefore embarrassingly parallel, and anything below ~99% is power/thermal sharing on
the 11.2 kW tray or measurement noise -- never interconnect.

## GPU specs

| Item | Value |
|---|---|
| Architecture | CDNA 4 (gfx950) |
| Compute units (per GPU) | 256 |
| Memory | 288 GB HBM3E |
| Memory bandwidth (per GPU) | 8 TB/s |
| PCIe host link | Gen 5 x16 (64 GB/s per direction) |
| GPU-GPU interconnect | Infinity Fabric (XGMI) 4th gen, ~1075 GB/s aggregate per GPU |
| TBP (per GPU) | 1400 W  (8 GPUs = 11.2 kW tray) |
| Driver / ROCm | amdgpu 6.19.14.31400100 / ROCm 7.14 |
| Host | 2 x EPYC 9575F (256 threads), 3.0 TiB RAM, Ubuntu 22.04.5 |

### Dense peak compute (per GPU, AMD published spec, no sparsity)

| Precision | Peak (TFLOPS) |
|---|---:|
| fp4 | 10,000.0 |
| fp6 | 10,000.0 |
| bf6 | 10,000.0 |
| fp8 | 5,000.0 |
| bf8 | 5,000.0 |
| fp16 | 2,500.0 |
| bf16 | 2,500.0 |
| fp32 | 157.3 |
| fp64 | 78.6 |

FP6/BF6 are block-scaled MX formats and run at the **FP4 rate** on CDNA 4 (10,000 TFLOPS), not half it.

## Measured TFLOPS (peak across log intervals)

Aggregate = sum of per-GPU peaks. Scaling = aggregate / N=1 value; perfect linear scaling would be N/1.

| Precision | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | 2x eff | 3x eff | 4x eff | 5x eff | 6x eff | 7x eff | 8x eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,975.7 | 7,901.6 | 11,024.5 | 14,990.3 | 13,615.2 | 16,566.4 | 16,890.3 | 17,733.2 | 1.99x (99%) | 2.77x (92%) | 3.77x (94%) | 3.42x (68%) | 4.17x (69%) | 4.25x (61%) | 4.46x (56%) |
| fp6 | 1,238.0 | 2,475.8 | 3,717.5 | 4,956.1 | 6,195.6 | 7,435.9 | 8,574.5 | 9,910.2 | 2.00x (100%) | 3.00x (100%) | 4.00x (100%) | 5.00x (100%) | 6.01x (100%) | 6.93x (99%) | 8.00x (100%) |
| bf6 | 1,238.6 | 2,475.0 | 3,716.6 | 4,953.3 | 6,196.7 | 7,437.3 | 8,642.3 | 9,908.8 | 2.00x (100%) | 3.00x (100%) | 4.00x (100%) | 5.00x (100%) | 6.00x (100%) | 6.98x (100%) | 8.00x (100%) |
| fp8 | 3,564.1 | 7,373.1 | 10,941.3 | 14,440.9 | 18,129.8 | 21,733.6 | 25,412.5 | 28,816.5 | 2.07x (103%) | 3.07x (102%) | 4.05x (101%) | 5.09x (102%) | 6.10x (102%) | 7.13x (102%) | 8.09x (101%) |
| bf8 | 3,220.4 | 6,472.2 | 9,941.1 | 13,208.9 | 16,256.3 | 19,716.1 | 23,045.8 | 26,423.6 | 2.01x (100%) | 3.09x (103%) | 4.10x (103%) | 5.05x (101%) | 6.12x (102%) | 7.16x (102%) | 8.21x (103%) |
| fp16 | 1,521.8 | 3,062.2 | 4,592.8 | 6,150.2 | 7,684.5 | 9,163.1 | 10,704.3 | 12,301.5 | 2.01x (101%) | 3.02x (101%) | 4.04x (101%) | 5.05x (101%) | 6.02x (100%) | 7.03x (100%) | 8.08x (101%) |
| bf16 | 1,628.0 | 3,288.3 | 4,935.0 | 6,562.9 | 8,153.2 | 9,869.0 | 11,477.2 | 13,113.8 | 2.02x (101%) | 3.03x (101%) | 4.03x (101%) | 5.01x (100%) | 6.06x (101%) | 7.05x (101%) | 8.06x (101%) |
| fp32 | 152.8 | 306.8 | 460.0 | 614.1 | 767.1 | 920.4 | 1,073.8 | 1,228.6 | 2.01x (100%) | 3.01x (100%) | 4.02x (100%) | 5.02x (100%) | 6.02x (100%) | 7.03x (100%) | 8.04x (100%) |
| fp64 | 76.6 | 153.4 | 230.5 | 307.6 | 384.1 | 461.2 | 537.8 | 614.9 | 2.00x (100%) | 3.01x (100%) | 4.01x (100%) | 5.01x (100%) | 6.02x (100%) | 7.02x (100%) | 8.03x (100%) |

## Per-GPU average TFLOPS

| Precision | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,975.7 | 3,950.8 | 3,674.8 | 3,747.6 | 2,723.0 | 2,761.1 | 2,412.9 | 2,216.7 |
| fp6 | 1,238.0 | 1,237.9 | 1,239.2 | 1,239.0 | 1,239.1 | 1,239.3 | 1,224.9 | 1,238.8 |
| bf6 | 1,238.6 | 1,237.5 | 1,238.9 | 1,238.3 | 1,239.3 | 1,239.5 | 1,234.6 | 1,238.6 |
| fp8 | 3,564.1 | 3,686.5 | 3,647.1 | 3,610.2 | 3,625.9 | 3,622.3 | 3,630.4 | 3,602.1 |
| bf8 | 3,220.4 | 3,236.1 | 3,313.7 | 3,302.2 | 3,251.3 | 3,286.0 | 3,292.3 | 3,302.9 |
| fp16 | 1,521.8 | 1,531.1 | 1,530.9 | 1,537.6 | 1,536.9 | 1,527.2 | 1,529.2 | 1,537.7 |
| bf16 | 1,628.0 | 1,644.2 | 1,645.0 | 1,640.7 | 1,630.6 | 1,644.8 | 1,639.6 | 1,639.2 |
| fp32 | 152.8 | 153.4 | 153.3 | 153.5 | 153.4 | 153.4 | 153.4 | 153.6 |
| fp64 | 76.6 | 76.7 | 76.8 | 76.9 | 76.8 | 76.9 | 76.8 | 76.9 |

### Per-GPU spread at N=8 (die-to-die variation)

| Precision | min | max | spread |
|---|---:|---:|---:|
| fp4 | 1,469.2 | 3,952.0 | 62.8% |
| fp6 | 1,236.2 | 1,241.0 | 0.4% |
| bf6 | 1,235.3 | 1,241.6 | 0.5% |
| fp8 | 3,501.6 | 3,733.5 | 6.2% |
| bf8 | 3,191.6 | 3,388.3 | 5.8% |
| fp16 | 1,508.8 | 1,570.7 | 3.9% |
| bf16 | 1,608.2 | 1,671.4 | 3.8% |
| fp32 | 152.6 | 154.3 | 1.1% |
| fp64 | 76.6 | 77.2 | 0.7% |

## Measured vs dense peak (per-GPU, N=1 run)

| Precision | Measured | Paper dense peak | % of peak |
|---|---:|---:|---:|
| fp4 | 3,975.67 | 10,000.0 | **39.8%** |
| fp6 | 1,238.05 | 10,000.0 | **12.4%** |
| bf6 | 1,238.56 | 10,000.0 | **12.4%** |
| fp8 | 3,564.06 | 5,000.0 | **71.3%** |
| bf8 | 3,220.37 | 5,000.0 | **64.4%** |
| fp16 | 1,521.82 | 2,500.0 | **60.9%** |
| bf16 | 1,627.98 | 2,500.0 | **65.1%** |
| fp32 | 152.82 | 157.3 | **97.2%** |
| fp64 | 76.62 | 78.6 | **97.5%** |

### Aggregate at N=8 vs aggregate peak

| Precision | Measured aggregate | Peak x8 | % of peak |
|---|---:|---:|---:|
| fp4 | 17,733.2 | 80,000.0 | 22.2% |
| fp6 | 9,910.2 | 80,000.0 | 12.4% |
| bf6 | 9,908.8 | 80,000.0 | 12.4% |
| fp8 | 28,816.5 | 40,000.0 | 72.0% |
| bf8 | 26,423.6 | 40,000.0 | 66.1% |
| fp16 | 12,301.5 | 20,000.0 | 61.5% |
| bf16 | 13,113.8 | 20,000.0 | 65.6% |
| fp32 | 1,228.6 | 1,258.4 | 97.6% |
| fp64 | 614.9 | 628.8 | 97.8% |

## Cross-machine comparison — Dell Cloud vs AMD Cloud vs B200 (per-GPU)

All three columns are per-GPU at N=1. **Dell Cloud and AMD Cloud are the same silicon** — 8 x MI355X (gfx950) — so the delta between them is a *software* delta:

| | Dell Cloud | AMD Cloud (this host) |
|---|---|---|
| ROCm | 7.2.3-90 | **7.14** |
| Code objects | gfx942 alias (`HSA_OVERRIDE_GFX_VERSION=9.4.2`) | **native gfx950** |
| Container | Singularity + ext3 overlay | Docker |
| gst duration | ~60 s | 30 s |

B200 reference measurements (provided, per-GPU): 768 TFLOPS FP32†, 1493 TFLOPS BF16, 4103 TFLOPS FP8. † see the FP32/TF32 note below the table — this is not a like-for-like figure and its ratio column is deliberately not computed.

| Precision | Dell Cloud MI355X | AMD Cloud MI355X | B200 ref | AMD/Dell | AMD/B200 | MI355X peak | B200 peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,159.52 | 3,975.67 | - | **1.26x** | - | 10,000.0 | 9,000 |
| fp6 | 1,280.17 | 1,238.05 | - | **0.97x** | - | 10,000.0 | - |
| bf6 | 1,280.20 | 1,238.56 | - | **0.97x** | - | 10,000.0 | - |
| fp8 | 3,610.88 | 3,564.06 | 4,103 | **0.99x** | **0.87x** | 5,000.0 | 4,500 |
| bf8 | 3,238.62 | 3,220.37 | - | **0.99x** | - | 5,000.0 | 4,500 |
| fp16 | 1,534.56 | 1,521.82 | - | **0.99x** | - | 2,500.0 | 2,250 |
| bf16 | 1,639.78 | 1,627.98 | 1,493 | **0.99x** | **1.09x** | 2,500.0 | 2,250 |
| fp32 | 153.76 | 152.82 | 768† | **0.99x** | _not comparable†_ | 157.3 | 80 |
| fp64 | 77.02 | 76.62 | - | **0.99x** | - | 78.6 | 40 |

AMD Cloud vs Dell Cloud ranges from **0.97x** (`fp6`) to **1.26x** (`fp4`). Since the silicon is identical, any gain is attributable to the newer ROCm and to running native gfx950 code objects instead of the gfx942 alias — which is exactly why this host does not set `HSA_OVERRIDE_GFX_VERSION`.

### Why `fp4` alone gains 1.26x

Every other precision lands in a tight 0.97x-0.99x band around Dell Cloud's number — essentially reproduction, not improvement. `fp4` is the lone outlier, and it is also a low-variance, reproducible measurement here: 0% per-GPU spread at N=1, still under 1% at N=2. That combination — one precision moving, everything else static, and the mover being clean data rather than noise — points at a specific software cause rather than run-to-run variance:

- **`fp4` is the newest, least mature kernel path in hipBLASLt** among the precisions tested. MX-block-scaled FP4 has had far less tuning time than BF16/FP8/FP32, which is exactly where a difference between a gfx950-native build and a gfx942-emulated one (Dell Cloud's `HSA_OVERRIDE_GFX_VERSION=9.4.2`) would most plausibly show up — an emulation layer is more likely to cost performance on a codepath that hasn't been separately hand-tuned for the emulated target.
- This is inference from the pattern, not a profiled root cause: no kernel-level trace was captured to confirm gfx942-emulation overhead specifically. The counter-evidence worth weighing is that `fp6`/`bf6` share the same MX block-scaling mechanism and the same 10,000 TFLOPS peak class as `fp4`, yet show **no** such gain (0.97x, i.e. slightly *below* Dell Cloud) — so "MX format in general" is not the explanation; it would have to be something specific to the fp4 numeric path itself.
- Also see the scaling-efficiency finding below: `fp4` is simultaneously the only precision with severely non-uniform multi-GPU scaling on *this* host (N=8 per-GPU spread up to 63%, vs <1% for every other precision including fp6/bf6). A kernel path immature enough to gain unusually from native codegen is also a plausible place to find launch or scheduling instability under concurrent multi-GPU load — the two observations may share a cause even though neither proves the other.

**† FP32 vs TF32 — this is NOT an apples-to-apples comparison.** The 768 TFLOPS B200 figure cannot be IEEE FP32 — B200's IEEE FP32 dense peak is only ~80 TFLOPS, the number in the "B200 peak" column above. 768 is almost certainly **TF32 tensor** (NVIDIA's reduced-precision 19-bit format, run on the tensor cores), whereas MI355X's 152.8/157.3 TFLOPS is **true IEEE-754 FP32 on the vector ALUs**. The two numbers are two different data types on two different execution units. It is included in the table only so a reader does not mistake its absence for "not measured" — the ratio column is deliberately left as "not comparable" rather than computed, because a 5.0x-looking number here would actively mislead: MI355X's true-FP32 is not 5x slower than anything, it is simply not the same operation as B200's TF32 path. RVS `gst` has no TF32 config, so that path is unmeasured on either MI355X host and no side-by-side TF32 number exists.

Only BF16 and FP8 above are like-for-like B200 reference measurements.

<!-- BEGIN fp4-investigation-result (auto-generated, do not edit by hand) -->

### fp4 N>=5 scaling investigation — result

Follow-up to the finding above (§A.7 of `plan.md`): 3 repeats each at N=5,6,7,8, with concurrent `rocm-smi` clock/power sampling. Full detail: `results/fp4_investigation.md`.

| N | GPU-ids low in every repeat | Overlap | Verdict |
|---:|---|---:|---|
| 5 | {none} | 0% | inconsistent — likely non-deterministic/software-correlated |
| 6 | {none} | 0% | inconsistent — likely non-deterministic/software-correlated |
| 7 | {51771} | 20% | inconsistent — likely non-deterministic/software-correlated |
| 8 | {17010} | 14% | inconsistent — likely non-deterministic/software-correlated |

**Bottom line**: 0/4 tested GPU-counts showed a consistent (same-GPU-every-repeat) low performer. None of the tested GPU-counts showed a consistent low performer across repeats — the low performer moved between runs, which points at non-determinism in RVS's parallel `gst` launch or the fp4 kernel path under concurrent multi-GPU load, not a specific die.

<!-- END fp4-investigation-result -->

## Why fp4 / fp6 / bf6 land so far below peak

**Short version.** Not memory bandwidth — these GEMMs use only 1-8% of HBM. Three causes, in increasing severity:

1. **MX block scaling** (fp4, fp6, bf6 only) — an E8M0 scale per 32-element block is real work the theoretical peak ignores. Costs roughly the fp4 40% vs fp8 71% gap.
2. **FP6 is not byte-aligned** (4 values per 3 bytes) — cross-byte unpacking, and likely no native full-rate MFMA path. This is why **fp6 is absolutely slower than fp8 (0.35x) despite twice the nominal peak**.
3. **Kernel maturity** — but only partly: native gfx950 codegen improved fp4 by **26%** while fp6 did not move **at all** (0.97x). So fp4 is under-tuned; fp6 hits a structural floor that better codegen does not touch.

The rest of this section is the evidence for each.

### It is not memory bandwidth

For the `gst` shape (8192x8192x16384, ~2.20 TFLOP per GEMM), required HBM bandwidth at the measured rates is:

| Precision | Bytes/GEMM | Bandwidth needed | % of 8 TB/s HBM |
|---|---:|---:|---:|
| fp4 | 277 MB | 500 GB/s | 6.3% |
| fp6 / bf6 | 344 MB | 194 GB/s | 2.4% |
| fp8 | 403 MB | 653 GB/s | 8.2% |
| bf16 | 671 MB | 497 GB/s | 6.2% |
| fp64 | 2282 MB | 80 GB/s | 1.0% |

Every precision uses **1-8% of HBM bandwidth**. These GEMMs have arithmetic intensity in the thousands of FLOP/byte — they are firmly compute-bound. Memory bandwidth is conclusively not the limiter, so the answer lies in the kernels.

### Cause 1 — MX block scaling costs throughput (fp4, fp6, bf6)

Exactly the three low outliers carry `scale_a: block, scale_b: block` in their generated conf; fp8/bf8/fp16/bf16 do not. MX formats attach an E8M0 scale per 32-element block, and applying those scales is real work that the theoretical peak number does not account for. fp4 at **39.8%** vs fp8 at **71.3%** is roughly the size of that tax.

### Cause 2 — FP6 is not byte-aligned, and pays much more (fp6, bf6 only)

Block scaling alone cannot explain fp6, because fp4 and fp6 share the same mechanism *and the same 10,000 TFLOPS nominal peak*, yet differ 3.2x. The absolute cross-precision ratios are the tell:

| Comparison | Measured | Nominal peak ratio |
|---|---:|---:|
| fp4 / fp8 | 1.12x | 2.00x |
| **fp6 / fp8** | **0.35x** | 2.00x |
| **fp4 / fp6** | **3.21x** | 1.00x |

**fp6 is slower in absolute terms than fp8** — 1238 vs 3564 TFLOPS — despite nominally having twice the peak. A format cannot be 2x faster on paper and 3x slower in practice unless it is not running on the fast path at all.

The most likely mechanism is packing: fp4 is 2 values per byte (clean nibbles) and fp8 is 1 value per byte, but **fp6 is 4 values per 3 bytes** — not byte-aligned. Feeding packed 6-bit operands into the matrix engine requires cross-byte bit extraction, and if the MFMA instruction cannot consume packed FP6 natively the kernel must widen it first, at which point throughput is set by the wider format, not by FP6's nominal rate. Consistent with that, measured fp6 (1238) sits at 0.81x measured fp16 (1522) — roughly bf16-class throughput minus unpack overhead.

### Cause 3 — kernel maturity, and the evidence that separates it from the above

Comparing the two MI355X hosts isolates software from silicon. Dell Cloud ran ROCm 7.2.3 with the gfx942 alias; this host runs ROCm 7.14 with native gfx950 code objects. **Identical hardware.** Only one precision responded:

| Precision | AMD Cloud / Dell Cloud |
|---|---:|
| **fp4** | **1.26x** |
| fp6, bf6 | 0.97x |
| fp8, bf8, fp16, bf16, fp32, fp64 | 0.99x |

fp4 gained 26% from native gfx950 codegen while **fp6 did not move at all**. That asymmetry is informative in both directions: fp4's shortfall is partly a *tuning* problem (it improves when the compiler targets the real architecture), whereas fp6's shortfall is a *structural* floor that better codegen does not touch — consistent with Cause 2 rather than with immature tuning.

### Caveat on the fp6 peak figure

The 10,000 TFLOPS peak used for fp6/bf6 comes from AMD's claim that CDNA 4 processes FP6 at the FP4 rate (a stated differentiator vs competitors that run FP6 at FP8 rate). If that claim does not hold for this silicon/stack, the correct denominator would be 5,000 and fp6 would read **24.8%** rather than 12.4% of peak. Either way it is the worst precision measured, and either way fp6 being absolutely slower than fp8 is the anomaly worth explaining. This is flagged because the percentage — unlike the measured TFLOPS — depends on a vendor claim this benchmark cannot verify.

### What would settle it

None of the above is a profiled root cause. Confirming Cause 2 requires kernel-level inspection — `rocprof` on a single fp6 GEMM to see which MFMA variant is issued and whether an unpack/convert kernel precedes it, or hipBLASLt's heuristic log to see which algorithm it selects for `fp6_e3m2_r`. That is a worthwhile follow-up if low-precision throughput matters for a real workload.

## Observations (auto-generated)

- `fp4`: N=8 scaling efficiency **56%** -- below the ~95% expected for an embarrassingly-parallel GEMM. Power sharing on the 11.2 kW tray is the leading explanation.
- `fp4`: die-to-die spread at N=8 is **62.8%** (1,469.2-3,952.0 TFLOPS) -- per-die clock variation under sustained load.
- `fp6`: only **12.4%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.
- `bf6`: only **12.4%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.

## Reproducing

```bash
cd /home/amd/shaohao/amd-benchmarks/amd-cloud && source common/env.sh
cd work-rocmval && ./run_part_a.sh          # smoke -> sweep -> health -> analysis
$PY analyze_rvs.py $LOG_ROOT/rvs/sweep_* -o $BENCH_ROOT/results
```

## Source data

| What | Where |
|---|---|
| Raw rvs stdout, one per (N, precision) | `logs/rvs/sweep_*/<n>x_<prec>.log` |
| Generated gst confs | `logs/rvs/sweep_*/<n>x_<prec>.conf` |
| Per-run summary | `logs/rvs/sweep_*/summary.{csv,txt}` |
| Health modules | `logs/rvs/health_*/` |
| This table as CSV | `results/rvs_tflops.csv` |

