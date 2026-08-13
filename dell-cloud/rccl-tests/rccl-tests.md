# RCCL-tests sweep — isolating the non-power-of-2 collective cliff

Standalone collective-bandwidth probe to isolate whether the N=5/6/7 slowdown
observed in [summary-1.md](../megatron-lm/summary-1.md) and [summary-2.md](../megatron-lm/summary-2.md) lives
in RCCL itself or somewhere in the Megatron-LM stack on top of it.

Driver: [`run-rccl-tests.sh`](run-rccl-tests.sh) (does not touch `run.sh`).

## What it does

- **Same container + same env as `run.sh`** (xGMI, IB disabled,
  `HSA_OVERRIDE_GFX_VERSION=9.4.2`, etc.) so the comparison is apples-to-apples.
- **Sweeps `N_GPUS = 2..8`** (N=1 is meaningless for collectives) and runs both
  `all_reduce_perf` and `all_gather_perf` — the two collectives Megatron's
  distributed-optimizer path lives on (`all-grads-sync` and
  `params-all-gather`).
- **Five RCCL configs** matching the hypothesis list:
  - `default` — exactly what `run.sh` uses (`NCCL_ALGO=Ring,Tree`,
    `NCCL_PROTO=Simple,LL,LL128`, `RCCL_MSCCL_ENABLE=1`)
  - `tree` — force `NCCL_ALGO=Tree`
  - `ring` — force `NCCL_ALGO=Ring`
  - `no_mscll` — `RCCL_MSCCL_ENABLE=0` (in case the default MSCCL plan is
    only tuned for {2,4,8})
  - `proto_simple` — force `NCCL_PROTO=Simple` (drop LL / LL128)
- **Message size range 16 MiB → 8 GiB**, bracketing Megatron's bucket sizes
  (~512 MB – 4 GB per bucket, inferred from the 32.44 GB BF16 model and the §2
  timings in the summaries).
- **Auto-locates `rccl-tests`** binaries inside the SIF; falls back to
  `RCCL_TESTS_DIR=...` env override and prints a build-from-source one-liner
  if it can't find them.
- **Summary table** at `logs/rccl_tests_<stamp>/rccl_tests_summary.txt` with one
  row per (collective, config, N): max size and busbw at that size.

## How to run

```bash
# full sweep (5 configs × 2 collectives × 7 N = 70 runs, ~30 min ballpark)
bash work/run-rccl-tests.sh

# just confirm the cliff on the baseline first
CONFIGS=default bash work/run-rccl-tests.sh

# focus on the suspect arities only
GPU_COUNTS="4 5 6 7 8" bash work/run-rccl-tests.sh

# narrow to one collective + one config (fastest debug loop)
CONFIGS=tree COLLECTIVES=all_reduce bash work/run-rccl-tests.sh

# explicit binary path if auto-detect fails
RCCL_TESTS_DIR=/opt/rccl-tests/build bash work/run-rccl-tests.sh
```

If the binary isn't in the image, build it once:

```bash
singularity exec --rocm /home/v89592/shaohao/megatron-lm/megatron-lm.sif bash -lc '
  cd /tmp && git clone https://github.com/ROCm/rccl-tests &&
  cd rccl-tests && make MPI=0 HIP_HOME=/opt/rocm -j
'
export RCCL_TESTS_DIR=/tmp/rccl-tests/build
```

## How to read the result

The summary file lists `busbw_GB/s` (bus bandwidth) per (collective, config, N).
For an 8-way xGMI mesh, the algorithm-bandwidth ceiling at a power-of-2 N is
roughly `link_bw × (N-1)/N`. Two patterns to look for:

1. **Default config also cliffs at N=5/6/7.** Then the slowdown is in RCCL,
   independent of Megatron. The other configs tell you which knob recovers it
   (e.g. `tree` flat across all N → ring-fit is the problem; `no_mscll` flat
   across all N → an MSCCL plan is misfiring; nothing recovers → genuinely
   missing tuning for those arities on gfx950).
2. **Default config is smooth in rccl-tests but Megatron still cliffs.** Then
   the regression is above RCCL — likely in how the distributed optimizer
   schedules its buckets, or in a stream-ordering / overlap issue exposed only
   under the Megatron workload. Worth profiling with `rocprof` next.

If a config recovers the cliff, it's a one-env-var workaround for `run.sh` —
just set the matching `NCCL_*` / `RCCL_*` variables in its `CONTAINER_ENV`
block. If nothing does, the real fix is to author an MSCCL plan for N=5/6/7
on the MI355X 8-way xGMI topology (or wait for ROCm's tuning tables to catch
up to gfx950).

## Q: Would ROCmValidationSuite (RVS) show the same observation?

No — and the reason is informative.

**RVS doesn't actually run RCCL.** Its standard modules test layers below the
collective library:

- `pqt` (P2P qualification) — pairwise `hipMemcpy` over xGMI, raw link bandwidth
- `pebb` / `pbqt` — PCIe BAR bandwidth
- `gst` — GEMM stress (compute)
- `babel` — HBM bandwidth
- `iet` — sustained power / EDP
- `gpup` / `rcqt` / `peqt` — config and register checks

None of those go through RCCL's algorithm layer. `pqt` is the closest neighbor —
it walks GPU pairs and measures xGMI link bandwidth — but it's strictly
point-to-point, never a ring or tree.

**Why the cliff wouldn't show up.** The N=5/6/7 slowdown is a property of how
RCCL maps a *collective algorithm* (balanced double-ring, tree, MSCCL plan)
onto the topology for a given N. The underlying xGMI links are symmetric and
healthy at every N — that's the §2 evidence in summary-1/2 (`forward-compute`
and `backward-compute` rank spread < 14 ms, every link carries its share). RVS
measures exactly that healthy raw layer, so it would come back clean.

**Useful read of an RVS run:**

- RVS clean + rccl-tests cliff → **hardware fine, RCCL algorithm tuning is the
  gap**. That's the working hypothesis from the summaries.
- RVS dirty (one link slow, one GPU low GEMM) → root cause is below RCCL; the
  collective cliff is a downstream symptom, and rccl-tests results would be
  misleading until the hardware is fixed.

So they're complementary, not redundant: **RVS validates the floor; rccl-tests
probes the algorithm layer where the cliff actually lives.** If you want a
single "is the box healthy?" gate before re-running the Megatron sweep,
`rvs -c conf/pqt.conf` plus `gst.conf` is a fine 10-minute sanity check, but it
won't reproduce what you saw at N=5/6/7.
