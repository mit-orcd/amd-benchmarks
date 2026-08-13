#!/usr/bin/env bash
# PART C driver: GPU-scan gate -> microbench sweep -> Megatron llama2-7B -> report.
# Strictly sequential, like run_part_a.sh / run_part_b.sh. Designed to run under
# setsid+nohup and survive logout.
#
# The gate matters: run_gpu_scan.sh is ~5 min and exercises the exact CLI surface
# (`primus-cli direct -- benchmark gemm ... --output-file`) that the 1-hour sweep depends
# on. Those flags were inherited from the older v26.3/v25.9 images and have never been
# validated against v26.5 on this host, so a flag-drift failure should cost 5 minutes,
# not an hour. If the gate produces no output files, we stop.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
DRV=$LOG_ROOT/primus/part_c_$(date +%Y%m%d_%H%M%S); mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "PART C start (RUN_ID=$RUN_ID, driver log: $DRV)"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy — another benchmark is running."
  exit 1
fi
say "all 8 GPUs idle, proceeding"

OUT_HOST=$BENCH_ROOT/primus/sweep_out_$RUN_ID

# ---- stage 1: GPU scan gate (~5 min) --------------------------------------------
say "STAGE 1/4 GPU scan gate (gemm, N=1..8)"
./run_gpu_scan.sh "$RUN_ID" >"$DRV/gpu_scan.log" 2>&1
rc=$?
produced=$(ls "$OUT_HOST"/gemm_N*.md 2>/dev/null | wc -l)
say "gpu scan rc=$rc, produced $produced/8 gemm output files"
if [[ "$produced" -eq 0 ]]; then
  say "ABORT: gate produced no benchmark output — the CLI flags likely drifted in v26.5."
  say "       Not committing to the 1-hour sweep. Inspect: $LOG_ROOT/primus/gpu-scan-$RUN_ID/"
  tail -25 "$LOG_ROOT/primus/gpu-scan-$RUN_ID/scan_1gpu.log" 2>/dev/null | tee -a "$STATE"
  exit 1
fi
say "gate OK"

# ---- stage 2: microbench sweep (~1 h) -------------------------------------------
say "STAGE 2/4 microbench sweep (gemm, gemm-dense, gemm-deepseek, attention, rccl x N=1..8)"
./run_full_sweep.sh "$RUN_ID" >"$DRV/full_sweep.log" 2>&1
say "full sweep rc=$? -> $LOG_ROOT/primus/sweep-$RUN_ID"

# ---- stage 3: Megatron llama2-7B (1.5-3 h) --------------------------------------
say "STAGE 3/4 Megatron llama2-7B BF16, N=1..8"
./run_megatron.sh "$RUN_ID" >"$DRV/megatron.log" 2>&1
say "megatron rc=$? -> $LOG_ROOT/primus/sweep-$RUN_ID"

# ---- stage 4: report -------------------------------------------------------------
say "STAGE 4/4 generate_report.py"
$PY generate_report.py \
    "$LOG_ROOT/primus/sweep-$RUN_ID" \
    "$OUT_HOST" \
    "$REF_ROOT/megatron-lm/summary.md" \
    "$BENCH_ROOT/results/PRIMUS_REPORT.md" >"$DRV/report.log" 2>&1
say "report rc=$? -> $BENCH_ROOT/results/PRIMUS_REPORT.md"

{ echo; echo "RUN_ID=$RUN_ID"; echo "SWEEP_DIR=$LOG_ROOT/primus/sweep-$RUN_ID";
  echo "OUT_HOST=$OUT_HOST"; } >>"$STATE"
say "PART C DONE"
