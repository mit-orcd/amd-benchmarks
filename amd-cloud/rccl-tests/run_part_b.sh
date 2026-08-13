#!/usr/bin/env bash
# PART B driver: smoke -> all-collective sweep -> config sweep -> analysis -> plot.
# Strictly sequential, like run_part_a.sh. Designed to run under nohup and survive
# logout. Refuses to start if the GPUs are busy.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/rccl/part_b_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "PART B start (driver log: $DRV)"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy — another benchmark is running."
  exit 1
fi
say "all 8 GPUs idle, proceeding"

# ---- stage 1: smoke (~2 min) ----------------------------------------------------
say "STAGE 1/4 smoke: all_reduce N=8"
COLLECTIVES=all_reduce GPU_COUNTS=8 OUT_TAG=smoke ./run-rccl-all.sh >"$DRV/smoke.log" 2>&1
rc=$?
smoke_dir=$(ls -dt "$LOG_ROOT"/rccl/rccl_smoke_* 2>/dev/null | head -1)
busbw=$(awk '$1=="all_reduce"{print $4}' "$smoke_dir/rccl_summary.txt" 2>/dev/null | tail -1)
say "smoke rc=$rc busbw=${busbw:-none} GB/s dir=${smoke_dir:-none}"
if [[ "$rc" -ne 0 || -z "$busbw" || "$busbw" == "-" ]]; then
  say "ABORT: smoke produced no busbw — not committing to the full sweep. See $DRV/smoke.log"
  exit 1
fi
say "smoke OK"

# ---- stage 2: full collective sweep (45-90 min) ----------------------------------
say "STAGE 2/4 all-collective sweep: 10 collectives x N=2..8"
./run-rccl-all.sh >"$DRV/all.log" 2>&1
rc=$?
all_dir=$(ls -dt "$LOG_ROOT"/rccl/rccl_all_* 2>/dev/null | head -1)
say "all-collective sweep rc=$rc -> $all_dir"

# ---- stage 3: config sweep (30-45 min) -------------------------------------------
say "STAGE 3/4 config sweep: 5 configs x {all_reduce,all_gather} x N=2..8"
./run-rccl-configs.sh >"$DRV/configs.log" 2>&1
rc=$?
cfg_dir=$(ls -dt "$LOG_ROOT"/rccl/rccl_tests_* 2>/dev/null | head -1)
say "config sweep rc=$rc -> $cfg_dir"

# ---- stage 4: analysis + plot -----------------------------------------------------
say "STAGE 4/4 analysis"
$PY analyze_rccl.py "$all_dir" "$cfg_dir" -o "$BENCH_ROOT/results" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/rccl.{md,csv}"
$PY plot_rccl_busbw.py "$BENCH_ROOT/results/rccl.csv" "$BENCH_ROOT/results/rccl_busbw.png" >"$DRV/plot.log" 2>&1
say "plot rc=$? -> $BENCH_ROOT/results/rccl_busbw.png"

{ echo; echo "ALL_DIR=$all_dir"; echo "CFG_DIR=$cfg_dir"; echo "SMOKE_DIR=$smoke_dir"; } >>"$STATE"
say "PART B DONE"
