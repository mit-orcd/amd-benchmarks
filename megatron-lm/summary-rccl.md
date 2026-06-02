# rccl-tests Sweep — Result Summary

**Sources**
- `work/log.nccl-tests` — nohup driver log for the sweep, started `2026-06-01 16:29:55 CDT`.
- `work/logs/rccl_tests_20260601_162955/` — per-(coll, config, N) logs and `rccl_tests_summary.txt`.
- Driver: [`work/run-rccl-tests.sh`](run-rccl-tests.sh) (see [`rccl-tests.md`](rccl-tests.md), [`readme-rccl.md`](readme-rccl.md)).
- Hardware/software: 1 node × up to 8 × AMD Instinct MI355X (gfx950) inside `megatron-lm.sif` (ROCm 6.4.3, RCCL 2.22.3, librccl `/opt/rocm/lib/librccl.so.1`). Same env as `run.sh` — IB off, xGMI peer-to-peer, `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
- Sweep matrix:
  - **Collectives:** `all_reduce_perf`, `all_gather_perf`.
  - **GPU counts:** N ∈ {2, 3, 4, 5, 6, 7, 8}.
  - **Configs (env overrides):** `default`, `tree` (`NCCL_ALGO=Tree`), `ring` (`NCCL_ALGO=Ring`), `no_mscll` (`RCCL_MSCCL_ENABLE=0`), `proto_simple` (`NCCL_PROTO=Simple`).
  - **Message sizes:** 16 MiB → 8 GiB (powers of 2).
  - **Iters:** 20, warmup 5; single-process multi-GPU (`-g N`).

> For the N=5/6/7 cliff analysis (causal chain, topology context, mitigations, vendor/generation reach), see [summary-power2.md](summary-power2.md). This document focuses on the raw collective measurements.

---

## At a glance

Bus bandwidth at the largest probe (8 GiB message, in-place, `default` config):

| N | all_reduce busbw (GB/s) | all_gather busbw (GB/s) |
|--:|------------------------:|------------------------:|
| 2 |                   61.24 |                   61.03 |
| 3 |                   75.15 |                   72.12 |
| 4 |                  168.59 |                  160.46 |
| 5 |                   38.62 |                   35.65 |
| 6 |                   38.38 |                   34.89 |
| 7 |                   37.77 |                   34.62 |
| 8 |                  381.33 |                  373.04 |

All other configs (`ring`, `no_mscll`, `proto_simple`) land within 1–3 % of `default` at every (collective, N). `tree` is a special case for `all_reduce` — see §3.

---

## 1. all_reduce

### 1a. busbw at 8 GiB, all configs (GB/s)

| N | default | tree   | ring   | no_mscll | proto_simple |
|--:|--------:|-------:|-------:|---------:|-------------:|
| 2 |   61.24 |  28.39 |  61.01 |    60.93 |        61.19 |
| 3 |   75.15 |  12.48 |  75.13 |    74.81 |        74.87 |
| 4 |  168.59 |  41.52 | 169.25 |   165.19 |       168.32 |
| 5 |   38.62 |   7.58 |  38.05 |    38.50 |        38.00 |
| 6 |   38.38 |   7.93 |  38.78 |    38.66 |        37.92 |
| 7 |   37.77 |   8.18 |  37.99 |    37.75 |        38.31 |
| 8 |  381.33 | 108.18 | 385.83 |   382.22 |       385.71 |

### 1b. Per-size scaling (default config, out-of-place busbw, GB/s)

| size   | N=4    | N=5   | N=8    |
|--------|-------:|------:|-------:|
| 16 MiB | 144.15 | 36.73 | 216.92 |
| 64 MiB | 159.12 | 34.72 | 306.09 |
| 256 MiB | 150.02 | 37.81 | 371.01 |
| 1 GiB | 165.50 | 36.86 | 384.28 |
| 4 GiB | 165.95 | 37.89 | 380.17 |
| 8 GiB | 163.48 | 37.98 | 381.78 |

`default` saturates by 256 MiB at N=4 and N=8; N=5 is steady-state from the 16 MiB probe onward at ~37 GB/s — i.e. the per-rank bandwidth ceiling is **a fixed multiplier, not small-message overhead**.

### 1c. Per-rank algorithm bandwidth (algbw, default config, 8 GiB)

`busbw = algbw × 2·(N−1)/N` for all_reduce, so algbw = busbw × N / (2·(N−1)). One xGMI link ≈ 64 GB/s.

| N | busbw (GB/s) | per-rank algbw (GB/s) | implied parallel xGMI lanes |
|--:|-------------:|----------------------:|----------------------------:|
| 2 |        61.24 |                  61.2 | ~1 link/rank |
| 3 |        75.15 |                  56.4 | ~1 link/rank |
| 4 |       168.59 |                 112.4 | ~2 links/rank |
| 5 |        38.62 |                  24.1 | ~0.4 link/rank |
| 6 |        38.38 |                  23.0 | ~0.4 link/rank |
| 7 |        37.77 |                  22.0 | ~0.3 link/rank |
| 8 |       381.33 |                 217.9 | ~3.5 links/rank |

---

## 2. all_gather

### 2a. busbw at 8 GiB, all configs (GB/s)

| N | default | tree   | ring   | no_mscll | proto_simple |
|--:|--------:|-------:|-------:|---------:|-------------:|
| 2 |   61.03 |  61.05 |  60.90 |    60.82 |        61.05 |
| 3 |   72.12 |  71.21 |  71.57 |    71.58 |        72.19 |
| 4 |  160.46 | 160.01 | 154.97 |   156.11 |       158.07 |
| 5 |   35.65 |  35.60 |  35.68 |    35.60 |        35.56 |
| 6 |   34.89 |  35.29 |  35.27 |    35.10 |        34.91 |
| 7 |   34.62 |  34.43 |  34.44 |    34.77 |        34.40 |
| 8 |  373.04 | 364.97 | 366.38 |   373.31 |       372.07 |

### 2b. Per-size scaling (default config, out-of-place busbw, GB/s)

| size   | N=4    | N=5   | N=8    |
|--------|-------:|------:|-------:|
| 16 MiB | 124.70 | ~34   | ~210   |
| 64 MiB | 145.17 | ~34   | ~300   |
| 256 MiB | 149.17 | ~35   | ~365   |
| 1 GiB | 155.77 | ~35   | ~370   |
| 4 GiB | 158.51 | ~35   | ~373   |
| 8 GiB | 160.46 | 35.65 | 373.04 |

(Approximate values for N=5 and N=8 from the per-size table in the raw log; the qualitative pattern matches `all_reduce` — saturation by ~256 MiB at N=4 and N=8; flat ~35 GB/s at N=5.)

---

## 3. Config-comparison observations

1. **`default` ≡ `ring` ≡ `no_mscll` ≡ `proto_simple`** at every (collective, N). The four configs land within 1–3 % of each other across the full sweep, so:
   - RCCL's default algorithm choice on this fabric *is* ring (with whichever protocol it picks at runtime).
   - Turning MSCCL off (`RCCL_MSCCL_ENABLE=0`) changes nothing — for these two collectives on `K₈` xGMI, MSCCL is either inactive or behaviorally identical to ring.
   - Forcing the protocol (`NCCL_PROTO=Simple`) changes nothing — LL/LL128 vs Simple don't move the bandwidth ceiling at 8 GiB.

2. **`tree` is the only knob with a real effect — and it is uniformly negative for `all_reduce`.** At every N, forcing `NCCL_ALGO=Tree` cuts all_reduce busbw to roughly half or less of ring (N=8: 381 → 108 GB/s; N=2: 61 → 28 GB/s; N=3: 75 → 12 GB/s — a 6× regression). For `all_gather`, tree results match ring within noise because RCCL's all_gather does not have a real tree implementation — every config produces the same numbers.

3. **No env-only configuration recovers the ~37 GB/s ceiling that all_reduce hits at N=5/6/7 or the ~35 GB/s ceiling that all_gather hits.** All four "non-tree" configs sit on the same floor at those N values.

---

## 4. Cross-check against Megatron-LM timers

The Megatron sweep ([summary-1.md](summary-1.md) / [summary-2.md](summary-2.md)) reports per-rank collective times at iter 45. Comparing the N=4 → N=5 step:

| layer                        | N=4 → N=5 ratio | source |
|------------------------------|----------------:|--------|
| Megatron `all-grads-sync`    | **4.06×** (334.5 → 1,354.8 ms) | summary-2 §2 |
| Megatron `params-all-gather` | **4.67×** (151.3 → 706.4 ms)   | summary-2 §2 |
| rccl-tests all_reduce busbw  | **4.37× slower** (168.59 → 38.62 GB/s) | this sweep |
| rccl-tests all_gather busbw  | **4.50× slower** (160.46 → 35.65 GB/s) | this sweep |

The Megatron application-level slowdown at every N matches the rccl-tests bandwidth ratio within ~10 %. Megatron is faithfully exposing what RCCL delivers underneath; nothing in the trainer is amplifying or generating the slowdown.

---

## 5. Other notable items

- **RCCL warnings (uniform across the sweep).** `NUMA auto balancing enabled` and `Missing iommu=pt` appear at every run. They add jitter to absolute numbers but are present at every N and config, so they do not explain any (collective, N) gap. Silencing requires root: `sudo sysctl kernel.numa_balancing=0` and adding `iommu=pt` to GRUB cmdline.
- **rccl-tests was built without gfx950 code objects** (offload-archs gfx906/908/90a/942 + gfx10xx). Same `HSA_OVERRIDE_GFX_VERSION=9.4.2` workaround as Megatron; the RCCL library itself (`librccl.so.1`) is what matters for collective performance, and that ships with the image — `RCCL version : 2.22.3-HEAD:7d8d67c+`.
- **Single-process driver (`-g N`).** rccl-tests was run with one CPU process driving N GPUs via HIP streams (not MPI). Megatron uses true multi-process distributed and shows matching per-N bandwidth ratios, so the single-process mode is not introducing the numbers — it's faithfully measuring RCCL's collective ceiling.
- **Memory probe sizes.** rccl-tests was driven up to 8 GiB messages to bracket Megatron's bucket sizes (~32 GB of grads sharded across DP ranks → per-collective payloads in the GiB range). Both collectives are saturated by ~256 MiB at N ∈ {2,3,4,8}, so the 8 GiB headline numbers represent the steady-state ceiling.
- **Two collectives, not five.** This sweep covers `all_reduce` and `all_gather` only — the two operations Megatron's distributed optimizer puts on the critical path (reduce-scatter is implemented via all_reduce + scatter in the current path, and the optimizer's parameter sync uses all_gather). `reduce_scatter`, `alltoall`, `broadcast`, `reduce`, `scatter`, `gather`, `sendrecv` binaries are all built in `rccl-tests/build/` but were not exercised here.

---

## Recommended next experiments

1. **Extend the coverage to `reduce_scatter` and `alltoall_perf`.** Both binaries are already built. `reduce_scatter` is the natural primitive behind distributed-optimizer grad sync and would let us see whether the same per-N pattern reproduces without going through the all_reduce path. `alltoall` would matter once MoE / TP-with-sequence-parallel is exercised.
2. **Multi-node.** The current sweep stays on xGMI (`NCCL_IB_DISABLE=1`). Rerun the same driver under multi-node `mpirun` to characterize the IB-crossing collective bandwidth, which becomes the limiter once a step crosses a node boundary.
3. **Re-run when ROCm/RCCL ships a gfx950 tuning update.** The same driver and summary script will produce a directly comparable table.
4. **File `rccl_tests_summary.txt` upstream** at github.com/ROCm/rccl as a clean repro — same hardware, same RCCL version, only (collective, config, N) varies.
