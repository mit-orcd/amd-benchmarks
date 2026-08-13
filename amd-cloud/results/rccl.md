# RCCL Collective Communications — MI355X x8, XGMI

System: 8 x AMD Instinct MI355X (gfx950), ROCm 7.14, XGMI all-to-all (K8 mesh, every pair 1 hop). Built natively for gfx950 with no `HSA_OVERRIDE_GFX_VERSION`, so absolute numbers may exceed the Dell Cloud gfx942-override run.

Source runs: rccl_all_20260813_211021, rccl_tests_20260813_213348

`busbw` is steady-state bytes crossing the wire per unit time, normalized for each algorithm's theoretical data movement -- the comparable metric across N and across collectives. All figures below are busbw at the top message size.

## 1. Measured results

### 1.1 Full collective sweep

| collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.6 | 90.4 | 167.6 | 45.8 | 44.5 | 43.8 | 388.1 | **79% down** |
| all_reduce | 59.6 | 92.4 | 169.5 | 48.0 | 47.7 | 47.6 | 396.3 | **77% down** |
| alltoall | 58.2 | 116.2 | 154.1 | 61.4 | 61.5 | 62.1 | 346.4 | **67% down** |
| alltoallv | 58.3 | 88.5 | 114.3 | 45.4 | 41.6 | 40.4 | 212.2 | **69% down** |
| broadcast | 62.0 | 88.9 | 175.0 | 39.6 | 39.4 | 39.3 | 389.6 | **81% down** |
| gather | 61.6 | 122.3 | 182.8 | 78.3 | 73.4 | 79.2 | 426.0 | **67% down** |
| reduce | 62.1 | 99.3 | 170.9 | 45.7 | 45.8 | 45.5 | 330.4 | **76% down** |
| reduce_scatter | 57.8 | 81.4 | 166.6 | 46.3 | 47.2 | 47.8 | 387.3 | **77% down** |
| scatter | 61.4 | 122.5 | 180.0 | 69.4 | 74.5 | 73.6 | 396.6 | **67% down** |
| sendrecv | 58.4 | 60.9 | 60.4 | 60.3 | 60.6 | 60.3 | 60.2 | none |

### 1.1a Dell Cloud vs AMD Cloud — full sweep (busbw GB/s at top message size)

Same silicon, same fabric (8 x MI355X, XGMI 4th gen K8 mesh) on both hosts; the only difference is software (ROCm 7.2.3 + gfx942 alias on Dell Cloud vs ROCm 7.14 native gfx950 here). Each cell is `Dell / AMD (AMD÷Dell)`.

| Collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|
| all_gather | 60.6/59.6 (0.98x) | 71.1/90.4 (1.27x) | 158.7/167.6 (1.06x) | 35.4/45.8 (1.29x) | 34.9/44.5 (1.28x) | 34.9/43.8 (1.26x) | 365.8/388.1 (1.06x) |
| all_reduce | 61.3/59.6 (0.97x) | 75.0/92.4 (1.23x) | 166.5/169.5 (1.02x) | 38.4/48.0 (1.25x) | 38.4/47.7 (1.24x) | 38.2/47.6 (1.25x) | 381.3/396.3 (1.04x) |
| alltoall‡ | 58.4/58.2 (1.00x) | 61.8/**116.2** (1.88x) | 155.2/154.1 (0.99x) | 44.1/61.4 (1.39x) | 45.6/61.5 (1.35x) | 44.3/62.1 (1.40x) | 360.9/346.4 (0.96x) |
| broadcast | 63.5/62.0 (0.98x) | 68.1/88.9 (1.31x) | 169.2/175.0 (1.03x) | 34.1/39.6 (1.16x) | 33.9/39.4 (1.16x) | 33.8/39.3 (1.16x) | 377.3/389.6 (1.03x) |
| gather | 72.1/61.6 (0.85x) | 78.3/**122.3** (1.56x) | 211.6/182.8 (0.86x) | 69.4/78.3 (1.13x) | 68.8/73.4 (1.07x) | 70.3/79.2 (1.13x) | 444.1/426.0 (0.96x) |
| reduce | 72.9/62.1 (0.85x) | 86.5/99.3 (1.15x) | 197.4/170.9 (0.87x) | 43.6/45.7 (1.05x) | 42.9/45.8 (1.07x) | 43.1/45.5 (1.06x) | 358.5/330.4 (0.92x) |
| reduce_scatter | 60.6/57.8 (0.95x) | 71.0/81.4 (1.15x) | 165.1/166.6 (1.01x) | 39.6/46.3 (1.17x) | 39.6/47.2 (1.19x) | 40.5/47.8 (1.18x) | 407.7/387.3 (0.95x) |
| scatter | 63.1/61.4 (0.97x) | 71.5/**122.5** (1.71x) | 191.6/180.0 (0.94x) | 65.3/69.4 (1.06x) | 65.6/74.5 (1.14x) | 66.4/73.6 (1.11x) | 426.4/396.6 (0.93x) |
| sendrecv | 59.2/58.4 (0.99x) | 60.3/60.9 (1.01x) | 60.6/60.4 (1.00x) | 43.8/60.3 (1.38x) | 43.8/60.6 (1.39x) | 43.4/60.3 (1.39x) | 53.2/60.2 (1.13x) |

‡ measured here at a smaller top message size than Dell Cloud's 8 GiB (both sides cap `alltoall`/`alltoallv` early to survive the N=5 OOM that killed Dell Cloud's alltoallv run — see run-rccl-all.sh `ALLTOALL_MAX`). busbw plateaus well before 8 GiB for every collective measured (Dell Cloud's own finding, summary-rccl.md §1.1), so the smaller cap should still land in the flat region, but it is not a strictly identical measurement and is flagged rather than presented as one.

Ratio ranges from **0.85x** (`reduce` N=2) to **1.88x** (`alltoall` N=3). Bold cells are >1.5x or <0.85x — outside what run-to-run noise on identical hardware would explain.

### 1.2 Infinity Fabric paper spec vs measured ceilings

Each GPU has 7 xGMI links wired point-to-point to the other 7 GPUs. On-node bandwidth telemetry reports N/A on this driver build, so these are AMD's published MI350-series peaks:

| Quantity | Spec |
|---|---:|
| Per xGMI link, bidirectional | **153.6 GB/s** |
| Per xGMI link, per direction | 76.8 GB/s |
| Per-GPU aggregate (x7 links), bidirectional | **1075.2 GB/s** |
| Per-GPU aggregate (x7 links), per direction | 537.6 GB/s |

The comparable ceiling depends on how many links the *specific* collective engages: sendrecv lights one link per pair, while a ring at N drives min(N, 7) links concurrently. Comparing every row to the full 7-link aggregate would understate small-N results.

| Collective | N | Measured (GB/s) | Ceiling (GB/s) | Basis | Achieved |
|---|---:|---:|---:|---|---:|
| all_gather | 2 | 59.80 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 2 | 59.61 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 3 | 90.40 | 230.4 | 3-link ring x 1 direction | 39% |
| all_gather | 3 | 90.39 | 230.4 | 3-link ring x 1 direction | 39% |
| all_gather | 4 | 167.86 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 4 | 167.55 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 5 | 45.80 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 5 | 45.84 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 6 | 44.47 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 6 | 44.53 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 7 | 43.72 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 7 | 43.80 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 8 | 389.20 | 537.6 | 7-link ring x 1 direction | 72% |
| all_gather | 8 | 388.10 | 537.6 | 7-link ring x 1 direction | 72% |
| all_reduce | 2 | 59.67 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 2 | 59.63 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 3 | 92.75 | 230.4 | 3-link ring x 1 direction | 40% |
| all_reduce | 3 | 92.41 | 230.4 | 3-link ring x 1 direction | 40% |
| all_reduce | 4 | 169.78 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 4 | 169.47 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 5 | 48.01 | 384.0 | 5-link ring x 1 direction | **13%** |
| all_reduce | 5 | 47.98 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_reduce | 6 | 47.68 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 6 | 47.66 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 7 | 47.59 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 7 | 47.64 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 8 | 396.58 | 537.6 | 7-link ring x 1 direction | 74% |
| all_reduce | 8 | 396.33 | 537.6 | 7-link ring x 1 direction | 74% |
| alltoall | 2 | 58.19 | 153.6 | 2 concurrent pairwise links | 38% |
| alltoall | 3 | 116.15 | 230.4 | 3 concurrent pairwise links | 50% |
| alltoall | 4 | 154.11 | 307.2 | 4 concurrent pairwise links | 50% |
| alltoall | 5 | 61.38 | 384.0 | 5 concurrent pairwise links | **16%** |
| alltoall | 6 | 61.47 | 460.8 | 6 concurrent pairwise links | **13%** |
| alltoall | 7 | 62.10 | 537.6 | 7 concurrent pairwise links | **12%** |
| alltoall | 8 | 346.42 | 537.6 | 7 concurrent pairwise links | 64% |
| alltoallv | 2 | 58.28 | 153.6 | 2 concurrent pairwise links | 38% |
| alltoallv | 3 | 88.45 | 230.4 | 3 concurrent pairwise links | 38% |
| alltoallv | 4 | 114.29 | 307.2 | 4 concurrent pairwise links | 37% |
| alltoallv | 5 | 45.38 | 384.0 | 5 concurrent pairwise links | **12%** |
| alltoallv | 6 | 41.61 | 460.8 | 6 concurrent pairwise links | **9%** |
| alltoallv | 7 | 40.36 | 537.6 | 7 concurrent pairwise links | **8%** |
| alltoallv | 8 | 212.21 | 537.6 | 7 concurrent pairwise links | 39% |
| broadcast | 2 | 61.97 | 153.6 | 2-link ring x 1 direction | 40% |
| broadcast | 3 | 88.88 | 230.4 | 3-link ring x 1 direction | 39% |
| broadcast | 4 | 175.02 | 307.2 | 4-link ring x 1 direction | 57% |
| broadcast | 5 | 39.57 | 384.0 | 5-link ring x 1 direction | **10%** |
| broadcast | 6 | 39.36 | 460.8 | 6-link ring x 1 direction | **9%** |
| broadcast | 7 | 39.31 | 537.6 | 7-link ring x 1 direction | **7%** |
| broadcast | 8 | 389.56 | 537.6 | 7-link ring x 1 direction | 72% |
| gather | 2 | 61.56 | 153.6 | 2 concurrent pairwise links | 40% |
| gather | 3 | 122.33 | 230.4 | 3 concurrent pairwise links | 53% |
| gather | 4 | 182.77 | 307.2 | 4 concurrent pairwise links | 59% |
| gather | 5 | 78.28 | 384.0 | 5 concurrent pairwise links | **20%** |
| gather | 6 | 73.38 | 460.8 | 6 concurrent pairwise links | **16%** |
| gather | 7 | 79.17 | 537.6 | 7 concurrent pairwise links | **15%** |
| gather | 8 | 426.03 | 537.6 | 7 concurrent pairwise links | 79% |
| reduce | 2 | 62.08 | 153.6 | 2-link ring x 1 direction | 40% |
| reduce | 3 | 99.28 | 230.4 | 3-link ring x 1 direction | 43% |
| reduce | 4 | 170.92 | 307.2 | 4-link ring x 1 direction | 56% |
| reduce | 5 | 45.67 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce | 6 | 45.80 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce | 7 | 45.51 | 537.6 | 7-link ring x 1 direction | **8%** |
| reduce | 8 | 330.35 | 537.6 | 7-link ring x 1 direction | 61% |
| reduce_scatter | 2 | 57.76 | 153.6 | 2-link ring x 1 direction | 38% |
| reduce_scatter | 3 | 81.38 | 230.4 | 3-link ring x 1 direction | 35% |
| reduce_scatter | 4 | 166.62 | 307.2 | 4-link ring x 1 direction | 54% |
| reduce_scatter | 5 | 46.34 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce_scatter | 6 | 47.19 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce_scatter | 7 | 47.78 | 537.6 | 7-link ring x 1 direction | **9%** |
| reduce_scatter | 8 | 387.34 | 537.6 | 7-link ring x 1 direction | 72% |
| scatter | 2 | 61.37 | 153.6 | 2 concurrent pairwise links | 40% |
| scatter | 3 | 122.47 | 230.4 | 3 concurrent pairwise links | 53% |
| scatter | 4 | 179.98 | 307.2 | 4 concurrent pairwise links | 59% |
| scatter | 5 | 69.37 | 384.0 | 5 concurrent pairwise links | **18%** |
| scatter | 6 | 74.49 | 460.8 | 6 concurrent pairwise links | **16%** |
| scatter | 7 | 73.63 | 537.6 | 7 concurrent pairwise links | **14%** |
| scatter | 8 | 396.63 | 537.6 | 7 concurrent pairwise links | 74% |
| sendrecv | 2 | 58.44 | 76.8 | 1 link x 1 direction | 76% |
| sendrecv | 3 | 60.91 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 4 | 60.42 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 5 | 60.34 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 6 | 60.63 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 7 | 60.27 | 76.8 | 1 link x 1 direction | 78% |
| sendrecv | 8 | 60.22 | 76.8 | 1 link x 1 direction | 78% |

Rows below 25% of their ceiling are bolded: at that level the arity is not constructing a usable communication pattern, rather than merely running inefficiently.

### 1.3 Interconnect comparison — Dell Cloud vs AMD Cloud vs NVIDIA reference

Dell Cloud and AMD Cloud are the **same fabric on the same silicon** (8 x MI355X, XGMI 4th gen, K8 direct mesh); only the software stack differs. The NVIDIA rows are **published spec only** — no NCCL run exists on either machine in this repo, so quoting someone else's busbw beside ours would not be like-for-like.

| Machine | Fabric | Topology | Per-link (bidir) | Per-GPU aggregate (bidir) | Per-GPU (per direction) | Measured AllReduce N=8 | % of ceiling |
|---|---|---|---|---:|---:|---:|---:|
| Dell Cloud — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | 381.27 GB/s | 71% |
| **AMD Cloud (this host)** — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | **396.33 GB/s** | 74% |
| NVIDIA H100 SXM (ref) — 8x GPU | NVLink 4 + NVSwitch | switched all-to-all | 25 GB/s x 18 links | 900.0 GB/s | 450.0 GB/s | _not measured (spec only)_ | — |
| NVIDIA B200 SXM (ref) — 8x GPU | NVLink 5 + NVSwitch | switched all-to-all | 50 GB/s x 18 links | 1800.0 GB/s | 900.0 GB/s | _not measured (spec only)_ | — |

Reading:

- **B200's NVLink 5 has ~1.67x the per-GPU fabric bandwidth of MI355X's XGMI** (1800 vs 1075 GB/s bidirectional). H100's NVLink 4 is slightly *below* MI355X (900 GB/s) — the dell-cloud readme's "comparable to NVLink 4" characterisation is right for H100 and wrong for B200.
- **The architectural difference matters more than the headline number.** NVIDIA routes through an NVSwitch, so any subset of GPUs gets full switched all-to-all bandwidth. AMD's mesh is direct point-to-point, which is why ring construction — and therefore the collective arity N — determines how much of the fabric is reachable.
- That is the root of the non-power-of-2 cliff: dell-cloud measured ~38 GB/s at N=5/6/7 (~7% of ceiling) for AllReduce, against 381 GB/s at N=8. A switched fabric has no equivalent failure mode, which is why NVIDIA stopped seeing these cliffs after DGX-1/P100 and why AMD's structural fix is UALink in MI400 rather than more MSCCL plans.

### 1.4 Same-silicon comparison vs Dell Cloud

Both hosts are 8 x MI355X. Dell Cloud ran ROCm 7.2.3 with the gfx942 alias; this host runs ROCm 7.14 with native gfx950 code objects.

| Collective | N | Dell Cloud | AMD Cloud | AMD/Dell |
|---|---:|---:|---:|---:|
| all_reduce | 4 | 166.48 | 169.47 | **1.02x** |
| all_reduce | 8 | 381.27 | 396.33 | **1.04x** |
| gather | 8 | 444.15 | 426.03 | **0.96x** |
| reduce_scatter | 8 | 407.69 | 387.34 | **0.95x** |
| scatter | 8 | 426.40 | 396.63 | **0.93x** |
| sendrecv | 2 | 59.21 | 58.44 | **0.99x** |

## 2. Config sweep — which knob recovers a cliff

### `all_gather` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.6 | 90.4 | 167.6 | 45.8 | 44.5 | 43.8 | 388.1 | **79% down** |
| no_mscll | 59.6 | 90.4 | 167.6 | 45.8 | 44.5 | 43.5 | 389.1 | **79% down** |
| proto_simple | 59.7 | 90.4 | 167.8 | 45.9 | 44.5 | 43.7 | 389.6 | **79% down** |
| ring | 59.7 | 90.4 | 168.1 | 45.9 | 44.5 | 43.4 | 388.9 | **79% down** |
| tree | 59.8 | 90.6 | 168.2 | 45.8 | 44.4 | 43.7 | 388.5 | **79% down** |

### `all_reduce` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.6 | 92.4 | 169.5 | 48.0 | 47.7 | 47.6 | 396.3 | **77% down** |
| no_mscll | 59.8 | 92.5 | 169.3 | 47.9 | 47.7 | 47.4 | 396.8 | **77% down** |
| proto_simple | 59.5 | 92.8 | 169.7 | 48.0 | 47.7 | 47.6 | 397.0 | **77% down** |
| ring | 59.8 | 92.7 | 169.7 | 48.0 | 47.7 | 47.4 | 396.3 | **77% down** |
| tree | 27.4 | 26.3 | 65.7 | 15.3 | 15.3 | 16.3 | 167.0 | **82% down** |

## 3. How to read the cliff column

`X% down` means the worst non-power-of-2 N is that much below the mean of the power-of-2 Ns.

- If the `default` config cliffs but `tree`/`ring`/`no_mscll` do not, the recovering knob is a one-env-var workaround and should be adopted.
- If nothing recovers it, the gap is missing RCCL tuning for those arities on gfx950 — RCCL has failed to construct a valid ring, and the fix is upstream.
- Either way the attribution to the *algorithm layer* rests on Part A's RVS `pbqt` (peer-to-peer XGMI) and `pebb` (PCIe) runs coming back clean. Without that, a cliff could equally be a bad link.

This matters for training: a Ring AllReduce at N=8 is the realistic upper bound for data-parallel gradient sync on this box — no Megatron dist-opt tuning can exceed it.

## 4. Reference

### 4.1 RCCL algorithm selection

| Collective | Ring | Tree | PAT | MSCCL | Pairwise sendrecv |
|---|---|---|---|---|---|
| AllReduce | Default large | Default small (degraded on mesh) | — | If plan exists | — |
| AllGather | Default | Falls back to Ring | Available | If plan exists | — |
| ReduceScatter | Default | Falls back to Ring | Available | If plan exists | — |
| Broadcast | Default large | Default small | — | — | — |
| Reduce | Default large | Default small | — | — | — |
| Gather / Scatter | — | — | — | — | Default |
| AllToAll | — | — | — | If plan exists | Default |
| SendRecv | — | — | — | — | Direct |

- **NVLS** (in-network reduction) is not applicable on AMD until UALink ships. **PAT** is a switched-fabric path, not active on an xGMI mesh.
- Forcing `NCCL_ALGO=Tree` on AllGather / ReduceScatter is silently equivalent to Ring.
- MSCCL plans ship in `/opt/rocm/share/rccl/msccl-algorithms/`; everything without a plan falls through to Ring or pairwise sendrecv.

### 4.2 Configuration knobs

| Variable | Value used here | Effect |
|---|---|---|
| `NCCL_ALGO` | `Ring,Tree` | Algorithm pool. `Tree` only affects AllReduce / Broadcast / Reduce. |
| `NCCL_PROTO` | `Simple,LL,LL128` | Wire protocol; `Simple` is effectively the default at large message size. |
| `RCCL_MSCCL_ENABLE` | `1` | Toggle MSCCL plan dispatch. |
| `NCCL_P2P_DISABLE` | `0` | `1` forces host-SHM staging; debug only. |
| `NCCL_SHM_DISABLE` | `0` | Leave on. |
| `NCCL_IB_DISABLE` | `1` | Single-node run, no IB. |
| `NCCL_SOCKET_IFNAME` | `lo` | Bootstrap over loopback. |
| `NCCL_DEBUG` | `WARN` | `INFO` prints algorithm + channel count per call. |
| `HSA_OVERRIDE_GFX_VERSION` | **unset** | Deliberately native gfx950; the gfx942 alias would undercount. |

### 4.3 Process model caveat

rccl-tests runs **one process driving N GPUs** (`-g N`, `MPI=0`), while real training and Primus' `benchmark rccl` run **N processes with 1 GPU each**. These take different code paths inside RCCL. If Part C's collective numbers disagree with these, the process model is the first suspect.

## 5. Source data

| What | Where |
|---|---|
| Raw rccl-tests stdout | `logs/rccl/rccl_*/<coll>_n<N>.log` |
| Config sweep logs | `logs/rccl/rccl_tests_*/<coll>_<cfg>_n<N>.log` |
| Per-run summary | `logs/rccl/rccl_*/rccl_summary.txt` |
| This table as CSV | `results/rccl.csv` |
| Figure | `results/rccl_busbw.png` |

