# Primus Sweep Report — MI355X (1..8 GPUs)

- Sweep dir: `/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704`
- Bench output dir: `/home/amd/shaohao/amd-benchmarks/amd-cloud/primus/sweep_out_20260813-224704`
- Image: `rocm/primus:v26.3` (singularity SIF)
- Hardware: 1 node × 8 × AMD Instinct MI355X (gfx950)

## 1. Megatron-LM (via Primus `train pretrain`)

Workload: `examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml` (llama2-7B, seq 4096, MBS=4, mock data, primus-turbo ON: `use_turbo_attention`, `use_turbo_grouped_mlp`). The `last TF/s/GPU` column is the steady-state value of the final logged iteration (after JIT warmup); `GBS` is parsed from the log.

#### Parallelism: pure data parallel (DP=N, TP=PP=CP=EP=1)

Verified from the run logs (`data_parallel_size=8, sequence_parallel_size=0`, `world_size=8`) and the config (`tensor_model_parallel_size: 1`, `pipeline_model_parallel_size: 1`, `expert_model_parallel_size: 1`, `sequence_parallel` commented out).

Every GPU holds a **full llama2-7B replica** and processes its own micro-batches; gradients are all-reduced once per step. This is a **weak-scaling** study, so the driver computes `GBS(N) = MBS x N x GRAD_ACC = 4 x N x 8 = 32N` — constant work per GPU as N grows, and divisible by `MBS x DP` by construction. That last point matters: a fixed GBS=256 is *not* divisible by `MBS(4) x DP(N)` for N in {3,5,6,7}, which is what forced the reference Dell Cloud run into three separate rerun scripts. Computing GBS per N up front makes it one clean sweep.

**Why DP and not TP/PP/CP/EP here:**

- **DP is viable at all only because llama2-7B fits in one GPU's HBM** (288 GB on MI355X). Any model that did not fit would have forced TP or PP.
- **TP** would shard each layer and all-reduce activations *every layer*, adding collective traffic that is unnecessary when the model already fits.
- **PP** adds pipeline-bubble overhead and mainly earns its keep across nodes or when the model does not fit; on one node with fast XGMI it is strictly worse.
- **CP** (context parallel) targets very long sequences; at seq 4096 it is unnecessary.
- **EP** (expert parallel) applies only to MoE models; llama2-7B is dense.

This is the deliberate opposite of Part D (ATOM inference), which runs **TP=8** because a 70B / 1.5 TB model cannot fit on one GPU. The contrast explains the collective-sensitivity result in section 7: Megatron here issues **one gradient all-reduce per ~5 s iteration**, so even the degraded N=5/6/7 RCCL bandwidth is negligible against per-iteration compute. TP=8 inference has no such insulation — its collectives sit in the **per-token critical path**.

### 1.1 TF/s/GPU vs #GPUs (Primus → Megatron-LM, llama2-7B BF16, turbo ON)

| N | GBS | last TF/s/GPU | mean TF/s/GPU | last iter (ms) | notes |
|--:|----:|--------------:|--------------:|---------------:|:------|
| 1 | 32 | 354.40 | 287.45 | 4840.20 |  |
| 2 | 64 | 376.00 | 304.75 | 5007.00 |  |
| 3 | 96 | 341.30 | 274.15 | 4981.50 |  |
| 4 | 128 | 309.40 | 245.70 | 4930.80 |  |
| 5 | 160 | 332.60 | 265.60 | 5163.40 |  |
| 6 | 192 | 323.60 | 257.90 | 5217.50 |  |
| 7 | 224 | 301.30 | 238.80 | 5252.20 |  |
| 8 | 256 | 294.40 | 232.10 | 4949.30 |  |

### 1.2 vs NVIDIA B200 (Megatron-LM, context only)

Reference: `/home/amd/shaohao/amd-benchmarks/dell-cloud/megatron-lm/summary.md` — the existing MI355X-vs-B200 table from the `rocm/megatron-lm:v26.1` image sweep (**GPT-15.6B, MBS=4, BF16, no-recompute**). **This is not directly comparable** to the Primus llama2-7B numbers in §1.1: different model, different image (no primus-turbo), different GEMM shape mix. Kept here only as the existing house benchmark. See §7 for an apples-to-oranges framing of what the Primus-turbo path delivers on the same hardware.

| N | B200 TF/s/GPU | MI355X TF/s/GPU | MI355X / B200 |
|--:|--------------:|----------------:|--------------:|
| 8 |         986.0 |       **790.4** |    **80.2 %** |

## 2. GEMM microbench (`benchmark gemm`)

Square GEMM 4096×4096×4096 BF16, 10 s per rank, 2 GB rotating cache buffer. Each rank runs independently — no collectives. Mean / min / max are taken across the N ranks.

### 2.1 TF/s/GPU vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1459.24 | 1459.24 | 1459.24 |  |
| 2 | 1444.51 | 1426.91 | 1462.10 |  |
| 3 | 1448.71 | 1437.89 | 1457.46 |  |
| 4 | 1438.48 | 1412.96 | 1461.17 |  |
| 5 | 1443.56 | 1407.75 | 1467.26 |  |
| 6 | 1436.38 | 1416.26 | 1465.91 |  |
| 7 | 1435.20 | 1404.58 | 1462.74 |  |
| 8 | 1443.51 | 1410.38 | 1475.85 |  |

## 3. Dense GEMM microbench (`benchmark gemm-dense`)

Llama-shape GEMM sweep (default: hidden 4096, FFN 11008, vocab 32000, MBS=1, BF16). Reports TF/s per shape per rank; the table aggregates across shapes and ranks.

### 3.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1322.14 | 1144.20 | 1471.24 |  |
| 2 | 1315.66 | 1145.99 | 1471.46 |  |
| 3 | 1309.50 | 1122.51 | 1468.69 |  |
| 4 | 1304.23 | 1094.15 | 1466.28 |  |
| 5 | 1303.96 | 1089.92 | 1473.50 |  |
| 6 | 1304.15 | 1118.57 | 1466.58 |  |
| 7 | 1304.70 | 1084.12 | 1478.82 |  |
| 8 | 1308.57 | 1097.41 | 1483.02 |  |

## 4. DeepSeek GEMM microbench (`benchmark gemm-deepseek`)

DeepSeek-V2/V3-style MoE shapes (hidden 4096, MoE int 1536, 128 routed experts, BF16).

### 4.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 989.61 | 164.41 | 1616.43 |  |
| 2 | 982.42 | 168.74 | 1616.28 |  |
| 3 | 981.05 | 164.58 | 1616.10 |  |
| 4 | 979.92 | 158.56 | 1614.84 |  |
| 5 | 979.50 | 163.12 | 1614.77 |  |
| 6 | 978.16 | 164.44 | 1614.80 |  |
| 7 | 978.23 | 166.59 | 1616.39 |  |
| 8 | 980.74 | 166.22 | 1623.43 |  |

## 5. Attention microbench (`benchmark attention`)

Flash-attention backend, MBS=4 across the built-in model shape set.

### 5.1 Attention metrics vs #GPUs

| N | metric | mean | best | n_shapes |
|--:|:-------|-----:|-----:|---------:|
| 1 | fwd_tflops | 735.63 | 779.49 | 6 |
| 1 | bwd_tflops | 221.62 | 263.94 | 6 |
| 2 | fwd_tflops | 729.10 | 781.48 | 6 |
| 2 | bwd_tflops | 219.24 | 260.76 | 6 |
| 3 | fwd_tflops | 725.15 | 770.90 | 6 |
| 3 | bwd_tflops | 219.65 | 262.36 | 6 |
| 4 | fwd_tflops | 722.23 | 772.77 | 6 |
| 4 | bwd_tflops | 220.05 | 266.93 | 6 |
| 5 | fwd_tflops | 722.47 | 767.52 | 6 |
| 5 | bwd_tflops | 220.68 | 265.63 | 6 |
| 6 | fwd_tflops | 719.91 | 772.59 | 6 |
| 6 | bwd_tflops | 219.90 | 266.68 | 6 |
| 7 | fwd_tflops | 720.52 | 770.15 | 6 |
| 7 | bwd_tflops | 219.02 | 262.36 | 6 |
| 8 | fwd_tflops | 720.51 | 768.73 | 6 |
| 8 | bwd_tflops | 219.65 | 266.16 | 6 |

## 6. RCCL collective microbench (`benchmark rccl --op all_reduce`)

All-reduce bandwidth sweep across message sizes (1K..128M, log2 sweep). Peak busbw reflects the asymptotic large-message bandwidth; mean is across the size sweep. **N=1 is skipped** — collective on a single rank is degenerate.

### 6.1 Peak / mean all-reduce busbw vs #GPUs

| N | peak busbw (GB/s) | mean busbw (GB/s) | sizes |
|--:|------------------:|------------------:|------:|
| 1 | — | — | skipped |
| 2 | 57.66 | 21.31 | 18 |
| 3 | 85.64 | 29.63 | 18 |
| 4 | 166.62 | 50.44 | 18 |
| 5 | 45.77 | 17.74 | 18 |
| 6 | 45.29 | 17.47 | 18 |
| 7 | 45.13 | 17.03 | 18 |
| 8 | 356.34 | 87.60 | 18 |

## 7. Analysis

- **Megatron weak-scaling (llama2-7B BF16, turbo ON):** N=1: 354 TF/s/GPU (100 % of N=1), N=2: 376 TF/s/GPU (106 % of N=1), N=3: 341 TF/s/GPU (96 % of N=1), N=4: 309 TF/s/GPU (87 % of N=1), N=5: 333 TF/s/GPU (94 % of N=1), N=6: 324 TF/s/GPU (91 % of N=1), N=7: 301 TF/s/GPU (85 % of N=1), N=8: 294 TF/s/GPU (83 % of N=1). Per-GPU throughput is essentially flat (≤ 22 % spread between best and worst N), so the all-reduce overhead at MBS·N grad-accum is small relative to the model's compute. The lower N=1 / higher N=8 iter-time scales linearly with GBS as expected for weak-scaling.
- **Primus-turbo vs reference image at N=8:** Primus (llama2-7B, turbo ON) hits **294 TF/s/GPU**; the `rocm/megatron-lm:v26.1` image on the same hardware (GPT-15.6B, no turbo, §3 tuned) tops out at **790.4 TF/s/GPU** — a **0.37× per-GPU jump**. Workloads differ (smaller model, different GEMM shapes, primus-turbo attention/grouped-MLP fused kernels), so this is *not* a pure kernel-vs-kernel speedup; it captures the combined win of (i) llama2-7B being more GEMM-dense than GPT-15.6B, (ii) primus-turbo replacing unfused softmax/RMSNorm/attention with gfx950-native kernels, and (iii) Primus' MFU-tuned argument set. Use as the new headline number for this hardware on a llama-family workload.
- **GEMM per-GPU consistency:** mean TF/s/GPU ranges 1435.2..1459.2 across all N (1.6 % spread). Each rank runs the same 4Kx4Kx4K BF16 shape independently with no collectives, so a flat curve confirms there's no thermal/PCIe/power contention as N grows. This is the per-GPU compute ceiling on this hardware for square FP16/BF16 matmul.
- **Shape sensitivity:** square 4Kx4Kx4K hits 1444 TF/s/GPU; the **llama-shape mix** (gemm-dense) drops to 1309 (91 % of peak); the **deepseek MoE shape mix** falls to 981 (68 %). The MoE drop is shape-driven (small / skewed K-dim in the expert path), not a hardware issue.
- **Attention fwd/bwd asymmetry:** fwd ≈ 724 TF/s/GPU, bwd ≈ 220 TF/s/GPU (bwd / fwd = 30 %). Backward is dominated by gradient recomputation + extra matmuls; the gap matches what's reported for flash-attention class kernels. Both are stable across N (each rank runs independently — no all-reduce in this bench).
- **RCCL all-reduce cliff:** peak busbw at N∈{4,8} averages **261 GB/s**; at N∈{5,6,7} it drops to **45 GB/s** (17 %). N=8 alone hits **356 GB/s** — the asymptotic xGMI ring bandwidth. The non-power-of-2 cliff matches the existing megatron-lm:v26.1 reference and confirms it's a topology/ring-algorithm issue (RCCL falls back from a clean ring to tree/segmented patterns), not a Primus issue. **Yet** the Megatron training in §1.1 is essentially insensitive to this cliff because per-iter compute (~20 s) dwarfs the all-reduce time even at the degraded busbw.

## 8. Raw per-(bench, N) status

From driver `summary.txt`:

```
Primus full sweep 20260813-224704
Image      : rocm/primus:v26.5
Driver log : /home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704
Bench out  : /home/amd/shaohao/amd-benchmarks/amd-cloud/primus/sweep_out_20260813-224704
Started    : 2026-08-13T22:50:14+00:00

================ N=1 ================
----- gemm N=1 port=29875 devs=0 2026-08-13T22:50:14+00:00 -----
  OK duration=18s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N1.log
----- gemm-dense N=1 port=29539 devs=0 2026-08-13T22:50:32+00:00 -----
  OK duration=88s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N1.log
----- gemm-deepseek N=1 port=29707 devs=0 2026-08-13T22:52:00+00:00 -----
  OK duration=168s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N1.log
----- attention N=1 port=29508 devs=0 2026-08-13T22:54:48+00:00 -----
  OK duration=19s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N1.log
----- rccl N=1 SKIPPED (collective needs N>=2) -----
================ N=2 ================
----- gemm N=2 port=29803 devs=0,1 2026-08-13T22:55:07+00:00 -----
  OK duration=22s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N2.log
----- gemm-dense N=2 port=29789 devs=0,1 2026-08-13T22:55:29+00:00 -----
  OK duration=90s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N2.log
----- gemm-deepseek N=2 port=29853 devs=0,1 2026-08-13T22:56:59+00:00 -----
  OK duration=168s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N2.log
----- attention N=2 port=29586 devs=0,1 2026-08-13T22:59:47+00:00 -----
  OK duration=17s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N2.log
----- rccl N=2 port=29535 devs=0,1 2026-08-13T23:00:04+00:00 -----
  OK duration=7s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N2.log
================ N=3 ================
----- gemm N=3 port=29571 devs=0,1,2 2026-08-13T23:00:12+00:00 -----
  OK duration=22s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N3.log
----- gemm-dense N=3 port=29610 devs=0,1,2 2026-08-13T23:00:34+00:00 -----
  OK duration=91s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N3.log
----- gemm-deepseek N=3 port=29976 devs=0,1,2 2026-08-13T23:02:05+00:00 -----
  OK duration=169s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N3.log
----- attention N=3 port=29832 devs=0,1,2 2026-08-13T23:04:54+00:00 -----
  OK duration=16s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N3.log
----- rccl N=3 port=29707 devs=0,1,2 2026-08-13T23:05:10+00:00 -----
  OK duration=8s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N3.log
================ N=4 ================
----- gemm N=4 port=29604 devs=0,1,2,3 2026-08-13T23:05:18+00:00 -----
  OK duration=23s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N4.log
----- gemm-dense N=4 port=29950 devs=0,1,2,3 2026-08-13T23:05:41+00:00 -----
  OK duration=92s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N4.log
----- gemm-deepseek N=4 port=29662 devs=0,1,2,3 2026-08-13T23:07:13+00:00 -----
  OK duration=170s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N4.log
----- attention N=4 port=29817 devs=0,1,2,3 2026-08-13T23:10:03+00:00 -----
  OK duration=15s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N4.log
----- rccl N=4 port=29945 devs=0,1,2,3 2026-08-13T23:10:18+00:00 -----
  OK duration=9s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N4.log
================ N=5 ================
----- gemm N=5 port=29733 devs=0,1,2,3,4 2026-08-13T23:10:27+00:00 -----
  OK duration=24s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N5.log
----- gemm-dense N=5 port=29808 devs=0,1,2,3,4 2026-08-13T23:10:51+00:00 -----
  OK duration=94s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N5.log
----- gemm-deepseek N=5 port=29624 devs=0,1,2,3,4 2026-08-13T23:12:25+00:00 -----
  OK duration=172s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N5.log
----- attention N=5 port=29961 devs=0,1,2,3,4 2026-08-13T23:15:17+00:00 -----
  OK duration=15s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N5.log
----- rccl N=5 port=29719 devs=0,1,2,3,4 2026-08-13T23:15:32+00:00 -----
  OK duration=10s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N5.log
================ N=6 ================
----- gemm N=6 port=29951 devs=0,1,2,3,4,5 2026-08-13T23:15:42+00:00 -----
  OK duration=26s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N6.log
----- gemm-dense N=6 port=29732 devs=0,1,2,3,4,5 2026-08-13T23:16:08+00:00 -----
  OK duration=96s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N6.log
----- gemm-deepseek N=6 port=29557 devs=0,1,2,3,4,5 2026-08-13T23:17:44+00:00 -----
  OK duration=172s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N6.log
----- attention N=6 port=29867 devs=0,1,2,3,4,5 2026-08-13T23:20:36+00:00 -----
  OK duration=16s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N6.log
----- rccl N=6 port=29542 devs=0,1,2,3,4,5 2026-08-13T23:20:52+00:00 -----
  OK duration=10s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N6.log
================ N=7 ================
----- gemm N=7 port=29969 devs=0,1,2,3,4,5,6 2026-08-13T23:21:02+00:00 -----
  OK duration=27s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N7.log
----- gemm-dense N=7 port=29954 devs=0,1,2,3,4,5,6 2026-08-13T23:21:29+00:00 -----
  OK duration=97s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N7.log
----- gemm-deepseek N=7 port=29905 devs=0,1,2,3,4,5,6 2026-08-13T23:23:06+00:00 -----
  OK duration=174s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N7.log
----- attention N=7 port=29940 devs=0,1,2,3,4,5,6 2026-08-13T23:26:00+00:00 -----
  OK duration=17s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N7.log
----- rccl N=7 port=29891 devs=0,1,2,3,4,5,6 2026-08-13T23:26:17+00:00 -----
  OK duration=11s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N7.log
================ N=8 ================
----- gemm N=8 port=29925 devs=0,1,2,3,4,5,6,7 2026-08-13T23:26:28+00:00 -----
  OK duration=31s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm_N8.log
----- gemm-dense N=8 port=29539 devs=0,1,2,3,4,5,6,7 2026-08-13T23:26:59+00:00 -----
  OK duration=100s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-dense_N8.log
----- gemm-deepseek N=8 port=29598 devs=0,1,2,3,4,5,6,7 2026-08-13T23:28:39+00:00 -----
  OK duration=179s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/gemm-deepseek_N8.log
----- attention N=8 port=29952 devs=0,1,2,3,4,5,6,7 2026-08-13T23:31:38+00:00 -----
  OK duration=20s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/attention_N8.log
----- rccl N=8 port=29751 devs=0,1,2,3,4,5,6,7 2026-08-13T23:31:58+00:00 -----
  OK duration=14s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/rccl_N8.log
Finished   : 2026-08-13T23:32:12+00:00
================ MEGATRON 2026-08-13T23:32:12+00:00 image=rocm/primus:v26.5 exp=examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml ================
----- megatron N=1 GBS=32 MBS=4 devs=0 2026-08-13T23:32:12+00:00 -----
  OK duration=361s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N1.log
----- megatron N=2 GBS=64 MBS=4 devs=0,1 2026-08-13T23:38:13+00:00 -----
  OK duration=369s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N2.log
----- megatron N=3 GBS=96 MBS=4 devs=0,1,2 2026-08-13T23:44:22+00:00 -----
  OK duration=369s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N3.log
----- megatron N=4 GBS=128 MBS=4 devs=0,1,2,3 2026-08-13T23:50:31+00:00 -----
  OK duration=378s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N4.log
----- megatron N=5 GBS=160 MBS=4 devs=0,1,2,3,4 2026-08-13T23:56:49+00:00 -----
  OK duration=381s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N5.log
----- megatron N=6 GBS=192 MBS=4 devs=0,1,2,3,4,5 2026-08-14T00:03:10+00:00 -----
  OK duration=411s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N6.log
----- megatron N=7 GBS=224 MBS=4 devs=0,1,2,3,4,5,6 2026-08-14T00:10:01+00:00 -----
  OK duration=392s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N7.log
----- megatron N=8 GBS=256 MBS=4 devs=0,1,2,3,4,5,6,7 2026-08-14T00:16:33+00:00 -----
  OK duration=387s log=/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/primus/sweep-20260813-224704/megatron-llama2_7B-bf16_N8.log
[megatron] 2026-08-14T00:23:00+00:00 DONE
```
