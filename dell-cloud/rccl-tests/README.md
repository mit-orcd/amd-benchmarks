# RCCL collective benchmarks — Dell Cloud server

Standalone RCCL collective-bandwidth work on 8 × MI355X (gfx950), run on the
**Dell Cloud** server. These files were originally filed under `../megatron-lm/` — they
share that suite's container and RCCL environment — but the subject here is RCCL itself,
not Megatron-LM training, so they now live in their own directory.

The question this suite exists to answer: the N=5/6/7 throughput cliff seen in the Megatron
sweeps — does it live in Megatron-LM, or in RCCL underneath it? Answer, from
[`summary-power2.md`](summary-power2.md): **in RCCL.**

## Documents

| File | What it is |
|------|------------|
| [`readme-rccl.md`](readme-rccl.md) | How to reproduce the sweeps: image, one-time `rccl-tests` build, the two driver scripts, env overrides |
| [`rccl-tests.md`](rccl-tests.md) | Design of the config sweep — the 5 RCCL configs, why each is tested, how to read the outcome. Also explains why RVS would *not* reproduce the cliff |
| [`summary-rccl.md`](summary-rccl.md) | Reference + analysis of every RCCL collective: bandwidth model, backing algorithms, Megatron's use of each, measured numbers |
| [`summary-power2.md`](summary-power2.md) | Result summary of the config sweep — the headline "the cliff is in RCCL, not Megatron" finding, plus mesh-vs-switched-fabric topology analysis (`summary-power2-bak.md` is an identical backup copy) |
| [`notes-amd.md`](notes-amd.md) | Companion note: why AMD shipped with this gap. Informed speculation, flagged as such in the file |
| `rccl_busbw_8GiB.png` | busbw-vs-N figure embedded in `summary-rccl.md §1.1` |

## Logs

| Directory | Sweep |
|-----------|-------|
| `logs/rccl_all_20260602_121713/` | All 10 collectives × N=2..8 (`summary-rccl.md §1.1`) |
| `logs/rccl_sendrecv_20260602_153246/` | sendrecv-only rerun, after the `alltoallv` N=5 OOM killed it mid-sweep |
| `logs/rccl_tests_20260601_162955/` | 5 configs × {all_reduce, all_gather} × N=2..8 (`summary-power2.md`) |
| `logs/rccl_tests_20260601_162747/` | First, aborted attempt at the above |

Each directory has a `*_summary.txt` with one row per (collective, config, N).

## Notes on paths and scripts

- The documents reference an on-host working directory of
  `/home/v89592/shaohao/megatron-lm/work/`, shared with the Megatron suite at the time of
  the run. Those paths are left as written — they record what was actually executed.
- The driver scripts (`run-rccl-all.sh`, `run-rccl-sendrecv.sh`, `run-rccl-tests.sh`,
  `plot_rccl_busbw.py`) are referenced by the docs but were **never committed**. Links to
  them in these files are dangling. Reimplementations against the same documented interface
  are specified in [`../../amd-cloud/plan.md`](../../amd-cloud/plan.md) Part B.
- Environment for all runs: `megatron-lm.sif` (ROCm 6.4.3, RCCL 2.22.3), IB off, xGMI
  peer-to-peer, `HSA_OVERRIDE_GFX_VERSION=9.4.2` (the image had no gfx950 code objects).

Cross-suite: the Megatron-LM sweeps that first surfaced the cliff are
[`../megatron-lm/summary-1.md`](../megatron-lm/summary-1.md) and
[`summary-2.md`](../megatron-lm/summary-2.md).
