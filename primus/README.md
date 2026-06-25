# Primus on MI355X — sweep + report harness

A reproducible benchmark harness for the AMD Primus framework
([AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus)) on a single
node of 8× AMD Instinct MI355X (gfx950). Sweeps GEMM / attention / RCCL
microbenches and Megatron-LM llama2-7B pretraining across N=1..8 GPUs, then
auto-generates `REPORT.md` with TF/s-vs-N tables and analysis.

## What's here

| Path | Purpose |
|------|---------|
| `Primus/` | Cloned `AMD-AIG-AIMA/Primus` repo (host-side, bound into v26.3 runs) |
| `image/primus-v26.3.sif` | Singularity image for MI300X/gfx942 (no primus-turbo) |
| `image/primus-v25.9_gfx950.sif` | Singularity image for MI355X/gfx950 (primus-turbo v0.1.0 prebuilt) |
| `overlay-megatron.img` | 20 GiB ext3 overlay (JIT kernel cache for the gfx950 image) |
| `run_full_sweep.sh` | All benches × N=1..8, using v26.3 image (no Megatron throughput) |
| `run_gpu_scan.sh` | Quick GEMM-only scan across 1..8 GPUs |
| `rerun_megatron_gfx950_v2.sh` | Megatron sweep using v25.9 gfx950 image (working throughput) |
| `rerun_megatron_missing.sh` | Rerun N∈{3,5,6,7} with valid `global_batch_size` |
| `rerun_megatron_gfx950.sh` | Initial gfx950 driver (pull + SIF convert + sweep) |
| `rerun_failed.sh` | Generic failed-bench rerun helper |
| `wait_and_report.sh` | Watch a sweep dir and regenerate REPORT.md when it finishes |
| `generate_report.py` | Parses logs + bench outputs into `REPORT.md` |
| `logs/sweep-<RUN_ID>/` | Per-bench, per-N driver logs + `summary.txt` |
| `Primus/sweep_out_<RUN_ID>/` | `.md` / `.csv` bench outputs (consumed by the report generator) |
| `scratch/` | tmp / triton / pip caches bound into v26.3 runs |
| `REPORT.md` | Latest generated report |

## Prerequisites

- Linux host with rootless **podman** and **singularity/apptainer ≥ 1.4**.
- 8× AMD Instinct MI355X (gfx950), ROCm driver, RCCL.
- **Python ≥ 3.11** on the host (for `generate_report.py` — uses 3.10+ syntax).
  On RHEL/CentOS the path is usually `/usr/bin/python3.11`.
- ≥ 50 GiB free disk under `image/` (SIF + overlay + JIT cache).

## Install

1. Clone Primus (only needed for v26.3 runs that bind-mount the repo):

   ```bash
   git clone https://github.com/AMD-AIG-AIMA/Primus
   ```

2. Pull the two container images via podman, then convert to SIF
   (singularity does not pull rootless `docker.io/` directly):

   ```bash
   mkdir -p image && cd image

   # gfx942 / MI300X image (general benches, no primus-turbo)
   podman pull docker.io/rocm/primus:v26.3
   podman save -o primus-v26.3.tar docker.io/rocm/primus:v26.3
   singularity build primus-v26.3.sif docker-archive://primus-v26.3.tar
   rm primus-v26.3.tar

   # gfx950 / MI355X image (Megatron-LM with primus-turbo built in)
   podman pull docker.io/rocm/primus:v25.9_gfx950
   podman save -o primus-v25.9_gfx950.tar docker.io/rocm/primus:v25.9_gfx950
   singularity build primus-v25.9_gfx950.sif docker-archive://primus-v25.9_gfx950.tar
   rm primus-v25.9_gfx950.tar
   ```

   The gfx950 image's bundled `/workspace/Primus` matches `primus_turbo
   v0.1.0`. Don't bind-mount the cloned `Primus/` over it — the API has
   drifted in HEAD and `PrimusTurboAttention` will fail to import.

3. Create the 20 GiB ext3 overlay for gfx950 Megatron runs (needed because
   `--writable-tmpfs` is only 16 MiB and the aiter / triton / primus-turbo
   JIT compile writes hundreds of MiB of `.cuda.o`):

   ```bash
   apptainer overlay create --size 20480 overlay-megatron.img
   ```

4. (Optional) For the v26.3 sweep, create scratch dirs that get bound into
   the container as `/tmp`, `/root/.triton`, `/root/.cache`:

   ```bash
   mkdir -p scratch/{tmp,triton,cache}
   ```

5. (Optional) For the B200 comparison table in `REPORT.md §1.2`, point the
   report generator at an existing `summary.md` from a `rocm/megatron-lm`
   sweep (see `generate_report.py` arg 3). If you don't have one, pass any
   empty file and §1.2 will be omitted.

## Run

All scripts hard-code their paths (sweep dir, image, overlay, output dir) at
the top — edit those before running on a different host.

### Quick smoke test — GEMM only

```bash
bash run_gpu_scan.sh
```

Writes to `logs/gpu-scan-<RUN_ID>/`. ~5 min.

### Full benchmark sweep (no Megatron throughput)

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
nohup bash run_full_sweep.sh "$RUN_ID" \
    > logs/sweep-${RUN_ID}/driver.out 2>&1 &
echo "$RUN_ID" > logs/CURRENT_RUN_ID.txt
```

Runs gemm / gemm-dense / gemm-deepseek / attention / rccl across N=1..8.
~1 h. Megatron entries in this sweep **will fail** on MI355X because
v26.3 lacks primus-turbo for gfx950 — use `rerun_megatron_gfx950_v2.sh`
to fill those in.

### Megatron-LM (the headline benchmark)

```bash
nohup bash rerun_megatron_gfx950_v2.sh \
    > logs/sweep-<RUN_ID>/megatron-gfx950.out 2>&1 &
```

Edit `RUN_ID=` at the top of the script to target the right `logs/sweep-*/`
directory. Runs N=1..8 sequentially using the gfx950 image + 20 GiB overlay.
The first run pays a ~10 min JIT compile tax; subsequent runs reuse the
overlay's cache (~5–10 min/N). Patches the model YAML in-place to use
`NullTokenizer` (no HF download needed) so `mock_data: true` works offline.

N=3/5/6/7 will fail on this sweep with
`AssertionError: global batch size (256) is not divisible by micro batch size (4) times data parallel size (N)`.
Fill those in with:

```bash
nohup bash rerun_megatron_missing.sh \
    > logs/sweep-<RUN_ID>/megatron-missing.out 2>&1 &
```

This patches `examples/megatron/configs/llama2_7B-pretrain.yaml` to set
`global_batch_size` to the nearest valid value (252 for N=3/7, 240 for
N=5/6), runs those 4 Ns, and restores GBS=256 in the overlay when done.

### Auto-regenerate on completion

`run_full_sweep.sh` doesn't call the report generator itself. Either:

- Run `wait_and_report.sh <RUN_ID>` alongside the sweep (watches `summary.txt`
  for "Finished", then runs `generate_report.py`), or
- Run `generate_report.py` manually after the sweep — see below.

The two `rerun_megatron_*.sh` scripts both regenerate `REPORT.md` automatically
when they finish.

## Analyze — `generate_report.py`

```bash
/usr/bin/python3.11 generate_report.py \
    logs/sweep-<RUN_ID> \
    Primus/sweep_out_<RUN_ID> \
    /path/to/b200/summary.md \
    REPORT.md
```

Inputs:

1. **Sweep dir** — contains per-(bench, N) driver logs (`<bench>_N<n>.log`,
   `megatron-llama2_7B-bf16_N<n>.log`) and `summary.txt`.
2. **Bench output dir** — contains the markdown/CSV files written by the
   benches themselves (`gemm_N<n>.md`, `attention_N<n>.csv`, etc.).
3. **B200 reference summary** — any markdown file containing a
   `| N | B200 TF/s/GPU | MI355X TF/s/GPU |` table; that block is lifted
   verbatim into `REPORT.md §1.2`. Pass an empty file to omit.
4. **Output path** — `REPORT.md` is overwritten in place.

What it extracts:

- **Megatron** (`parse_megatron`) — TF/s/GPU and iter ms from every
  `throughput per GPU` line in the log; `global batch size` parsed from
  the same line. The "last" value is the steady-state; the "mean" skips
  the first 2 warmup iters.
- **GEMM family** (`parse_gemm_like`) — handles both the plain `tflops`
  column in `benchmark gemm` output and the compound `X.XXs / YYY.YYTF/s /
  ZZZ.ZZGB/s / AI=…` cells in `gemm-dense` / `gemm-deepseek`.
- **Attention** — TFLOP-ish columns in the CSV (`fwd_tflops`, `bwd_tflops`).
- **RCCL** — `eff_gbps` (or `busbw` / `bw_gbps` / similar) column; reports
  peak and mean.

Section 7 ("Analysis") is computed from the parsed numbers — weak-scaling
efficiency, Primus-turbo vs reference-image speedup, GEMM shape sensitivity,
attention fwd/bwd ratio, RCCL non-power-of-2 cliff. Edit
`generate_report.py:main()` to change the narrative.

## Troubleshooting

- **`No space left on device` mid-Megatron run** — `--writable-tmpfs` is
  16 MiB; the gfx950 image needs the 20 GiB overlay (`overlay-megatron.img`).
  The `rerun_megatron_gfx950_v2.sh` and `rerun_megatron_missing.sh` scripts
  already use `--overlay`.
- **`primus_turbo not importable`** on a v25.9_gfx950 run — you bind-mounted
  the cloned `Primus/` over the image's `/workspace/Primus`. Don't; the API
  in HEAD doesn't match `primus_turbo v0.1.0` shipped in the image.
- **HF tokenizer download fails** in Megatron — the rerun scripts patch
  `tokenizer_type: Llama2Tokenizer` → `NullTokenizer` and add
  `vocab_size: 32000` in the overlay so `mock_data: true` works offline.
- **N=1 Megatron times out at 1800 s** — single-GPU iters are ~42 s, so
  50 iters + ~10 min JIT warmup ≈ 45 min. Raise the `timeout` in the
  script if you need all 50 iters; the steady-state TF/s is already
  reliable from the first ~10 captured iters.
- **RCCL `Unsupported op: all_reduce`** — Primus' argparse lists
  `all_reduce` but the backend expects `allreduce` (no underscore). Drop
  the `--op` flag to use the default.
