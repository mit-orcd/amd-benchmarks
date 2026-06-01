# rccl-tests Sweep — Result Summary

**Sources**
- `work/log.nccl-tests` — nohup driver log for the sweep, started `2026-06-01 16:29:55 CDT`.
- `work/logs/rccl_tests_20260601_162955/` — per-(coll, config, N) logs and `rccl_tests_summary.txt`.
- Driver: [`work/run-rccl-tests.sh`](run-rccl-tests.sh) (see [`rccl-tests.md`](rccl-tests.md), [`readme-rccl.md`](readme-rccl.md)).
- Hardware/software: 1 node × up to 8 × AMD Instinct MI355X (gfx950) inside `megatron-lm.sif` (ROCm 6.4.3, RCCL 2.22.3, librccl `/opt/rocm/lib/librccl.so.1`). Same env as `run.sh` — IB off, xGMI peer-to-peer, `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
- Sweep: collectives `all_reduce_perf`, `all_gather_perf`; sizes 16 MiB → 8 GiB ×2; iters=20 warmup=5; single-process multi-GPU (`-g N`).

The purpose of this sweep is to answer the open question from [summary-1.md](summary-1.md) and [summary-2.md](summary-2.md): is the N=5/6/7 collective cliff a Megatron-LM regression or does it live in the RCCL/xGMI layer below it?

---

## 1. Headline — the cliff is in RCCL, not Megatron  ★

At the largest probe (8 GiB message), bus bandwidth (busbw, in-place, default config):

| N | all_reduce busbw (GB/s) | all_gather busbw (GB/s) | vs N=8 |
|--:|------------------------:|------------------------:|------:|
| 2 |               **61.24** |               **61.03** |  0.16× |
| 3 |               **75.15** |               **72.12** |  0.20× |
| 4 |              **168.59** |              **160.46** |  0.44× |
| 5 |               **38.62** |               **35.65** | **0.10×** |
| 6 |               **38.38** |               **34.89** | **0.10×** |
| 7 |               **37.77** |               **34.62** | **0.09×** |
| 8 |              **381.33** |              **373.04** |  1.00× |

**The same cliff shape Megatron showed at the application level reproduces under raw RCCL.** N=5/6/7 collapse to ~38 GB/s for all_reduce (~35 GB/s for all_gather) while N=8 hits ~381 GB/s — a ~10× gap with no Megatron, no PyTorch, no distributed-optimizer scheduling in the picture. This rules out the Megatron-LM stack as the source of the regression.

Quantitative cross-check against summary-2 §2 (Megatron timers, iter 45):

| layer                       | N=4 → N=5 ratio | source |
|-----------------------------|----------------:|--------|
| Megatron `all-grads-sync` time | **4.06×** (334.5 → 1,354.8 ms) | summary-2 §2 |
| Megatron `params-all-gather` | **4.67×** (151.3 → 706.4 ms)  | summary-2 §2 |
| rccl-tests all_reduce busbw  | **4.37× slower** (168.59 → 38.62 GB/s) | this sweep |
| rccl-tests all_gather busbw  | **4.50× slower** (160.46 → 35.65 GB/s) | this sweep |

The slowdown magnitudes match within ~10 %. Megatron is faithfully exposing what RCCL is doing underneath; it isn't amplifying or generating the cliff.

---

## 2. None of the env-knob workarounds recover the cliff  ★

`all_reduce` busbw at 8 GiB, all configs:

| N | default | tree   | ring   | no_mscll | proto_simple |
|--:|--------:|-------:|-------:|---------:|-------------:|
| 2 |   61.24 | 28.39  |  61.01 |    60.93 |        61.19 |
| 3 |   75.15 | 12.48  |  75.13 |    74.81 |        74.87 |
| 4 |  168.59 | 41.52  | 169.25 |   165.19 |       168.32 |
| 5 |   38.62 |  7.58  |  38.05 |    38.50 |        38.00 |
| 6 |   38.38 |  7.93  |  38.78 |    38.66 |        37.92 |
| 7 |   37.77 |  8.18  |  37.99 |    37.75 |        38.31 |
| 8 |  381.33 | 108.18 | 385.83 |   382.22 |       385.71 |

`all_gather` busbw at 8 GiB (one column per config — all essentially identical, only `default` shown):

| N | 2     | 3     | 4      | 5     | 6     | 7     | 8      |
|--:|------:|------:|-------:|------:|------:|------:|-------:|
| busbw GB/s | 61.03 | 72.12 | 160.46 | 35.65 | 34.89 | 34.62 | 373.04 |

**Observations.**

1. **`default` ≡ `ring` ≡ `no_mscll` ≡ `proto_simple`** at every N. These four configs land within 1–3 % of each other across the whole sweep, which means RCCL's default algorithm choice on this fabric *is* ring (with whichever protocol it picks at runtime), and turning MSCCL off or forcing the protocol changes nothing. The hypothesis from summary-2 that "an MSCCL plan tuned only for {2,4,8} is misfiring" is **not supported** — MSCCL is already either inactive or behaviorally identical to ring here.
2. **`tree` is dramatically worse** for `all_reduce` at every N — N=8 drops from 381 to 108 GB/s, and N=5/6/7 fall to 7–8 GB/s. Forcing `NCCL_ALGO=Tree` is the *wrong* knob on this fabric; ring beats tree even where ring cliffs. (For `all_gather`, tree ≡ ring because RCCL's `all_gather` doesn't actually have a tree algorithm — every config produces the same numbers.)
3. **No combination tested moves N=5/6/7 off ~38 GB/s.** The cliff is intrinsic to the ring algorithm at non-power-of-2 sizes on the MI355X 8-way xGMI mesh. **An env-only mitigation does not exist in this configuration.**

---

## 3. The cliff is bandwidth-bound, not latency-bound

Per-size scaling at the all_reduce default config (out-of-place busbw, GB/s):

| size  | N=4   | N=5  | N=8   |
|-------|------:|-----:|------:|
| 16 MiB | 144.15 | 36.73 | 216.92 |
| 64 MiB | 159.12 | 34.72 | 306.09 |
| 256 MiB | 150.02 | 37.81 | 371.01 |
| 1 GiB | 165.50 | 36.86 | 384.28 |
| 4 GiB | 165.95 | 37.89 | 380.17 |
| 8 GiB | 163.48 | 37.98 | 381.78 |

At N=5 the algorithm is already saturated at ~37 GB/s at 16 MiB and stays flat all the way to 8 GiB. So the cliff is **not** small-message latency overhead amortizing badly — it is a steady-state bandwidth ceiling that the N=5/6/7 ring simply cannot exceed. That's consistent with the Megatron timers being a fixed multiplier worse, not a fixed offset worse.

---

## 4. Power-of-2 scaling is super-linear (and that's expected)

| N    | all_reduce busbw | per-rank algbw | implied parallel xGMI lanes |
|-----:|-----------------:|---------------:|----------------------------:|
| 2    |          61.24   | 61.24 / 1.00 = **61.2** | ~1 link/rank |
| 4    |         168.59   | 168.59 / 1.50 = **112.4** | ~2 links/rank |
| 8    |         381.33   | 381.33 / 1.75 = **217.9** | ~3.5 links/rank |

`busbw = algbw × 2·(N−1)/N` for all_reduce. The implied per-rank in-flight bandwidth grows roughly linearly with N at powers of 2 because RCCL constructs more parallel rings (channels) as it has more peers to chain together. xGMI on MI300X-class hardware has 7 links per die; at N=8 RCCL clearly engages several of them simultaneously, which is why N=8 is **2.27× faster than N=4** instead of just 2× (the "super-linear" effect noted in summary-1/2 §3 falls right out of this).

At N=5/6/7 the algorithm appears to drop to a single-channel ring (busbw drops to ~38 GB/s ≈ what one xGMI lane can sustain). That is the cliff's structural mechanism: the channel-construction step at non-power-of-2 N can't form multiple disjoint balanced rings on the 8-way mesh.

---

## 5. What this rules out / leaves open

**Ruled out** by this sweep:

- Megatron-LM is *not* the source of the cliff. The bucket-scheduling, distributed-optimizer, and compute/comm overlap logic in Megatron all behave the same way at N=5/6/7 as at N=4/8; the underlying collective is just slower.
- xGMI link health is *not* the source. N=2/3/4/8 all hit healthy per-link bandwidth (~30 GB/s per direction at N=2, scaling out cleanly). RCCL also reports `NUMA auto balancing` and `iommu=pt missing` warnings but those affect all N uniformly, not selectively the odd arities.
- MSCCL misfire is *not* the source — `RCCL_MSCCL_ENABLE=0` matches the default exactly.
- Protocol choice is *not* the source — `NCCL_PROTO=Simple` matches the default exactly.
- The naive "force `NCCL_ALGO=Tree`" workaround from summary-2's recommended-experiments list is *worse*, not better. Strike it from the list.

**Still on the table:**

- A genuinely custom MSCCL/MSCCL++ plan for N=5/6/7 on the 8-way MI355X xGMI mesh (multi-channel ring construction with a hand-chosen Hamiltonian decomposition). This is the only intervention I'd still expect to recover the cliff *without* changing N.
- A future RCCL release with gfx950-aware tuning tables. RCCL 2.22.3 is recent but its tuning was likely targeted at MI300X (gfx942); the MI355X mesh may need its own entries.
- Upstream-report this with an `rccl-tests` repro link. The numbers in §1 are clean enough to be filed against ROCm/rccl as-is.

**Practical takeaway for the Megatron sweep:**

For the existing single-node workload, **the only effective mitigation is N ∈ {2, 3, 4, 8}**. Avoid N=5/6/7 entirely until either a tuned MSCCL plan ships or RCCL grows MI355X-aware ring construction. summary-1/2 §3's "65 % efficiency at N=5/6/7" is not an artifact of the Megatron measurement — it's the actual collective ceiling at those arities on this hardware/software stack.

---

## Other notable items

- **RCCL warnings.** `NUMA auto balancing enabled` and `Missing iommu=pt` appear at every run. They could add jitter to absolute numbers but cannot explain the cliff (which is uniform across every config + reproducible run-to-run). Silencing requires root: `sudo sysctl kernel.numa_balancing=0` and adding `iommu=pt` to GRUB cmdline.
- **N=3 is consistently good.** It hits 75 GB/s for all_reduce — *higher* than N=2 (61 GB/s). A 3-ring is trivially balanced on any topology, so the cliff really is specific to N ∈ {5, 6, 7}, not "all non-power-of-2".
- **rccl-tests was built without gfx950 code objects** (offload-archs gfx906/908/90a/942 + gfx10xx). Same `HSA_OVERRIDE_GFX_VERSION=9.4.2` workaround as Megatron; the RCCL library itself (`librccl.so.1`) is what matters for collective performance, and that ships with the image — `RCCL version : 2.22.3-HEAD:7d8d67c+`.
- **Single-process driver (`-g N`).** rccl-tests was run with one CPU process driving N GPUs via HIP streams (not MPI). Megatron uses true multi-process distributed and shows the same cliff magnitude, so the single-process mode is not introducing the cliff — it's faithfully measuring RCCL's collective ceiling.

---

## Recommended next experiments

1. **File the cliff upstream** at github.com/ROCm/rccl with this `rccl_tests_summary.txt` as the repro. The data is clean: same hardware, same RCCL version, only N varies.
2. **Try MSCCL++ if available in the image.** MSCCL++ exposes a kernel-level GPU-driven collective path that bypasses NCCL's algorithm-selection step entirely. If the image has it, a hand-rolled `all_reduce` plan for N=5/6/7 is the next thing to try.
3. **Drop the "Tree-only" experiment** from summary-1/2 §recommended — this sweep already disproves it.
4. **Re-run when ROCm/RCCL ships a gfx950 tuning update.** The same script will produce a directly comparable table.
5. **Multi-node.** The current sweep stays on xGMI. The next interesting question is whether the analogous cliff exists at non-power-of-2 *node* counts once IB is in the path; reuse the same script with multi-node `mpirun` to find out.
