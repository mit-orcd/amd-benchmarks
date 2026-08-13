#!/usr/bin/env bash
# PART A driver: smoke -> gst sweep -> health modules -> analysis.
# Strictly sequential: every stage wants all 8 GPUs and the whole power envelope,
# so nothing here may overlap. Designed to run under nohup and survive logout.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/rvs/part_a_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt

say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "PART A start (driver log: $DRV)"
say "rvs: $RVS_BIN"

# ---- guard: refuse to start if the GPUs are busy -------------------------------
busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy — another workload is running"
  exit 1
fi
say "all 8 GPUs idle, proceeding"

# ---- stage 1: smoke test (~1 min) ----------------------------------------------
say "STAGE 1/4 smoke: 1 GPU x fp16 x 10s"
SMOKE=$LOG_ROOT/rvs/smoke_$TS
GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 OUT_DIR="$SMOKE" \
  ./run_tflops.sh >"$DRV/smoke.log" 2>&1
rc=$?
agg=$(awk -F, 'NR==2{print $(NF-1)}' "$SMOKE/summary.csv" 2>/dev/null)
say "smoke rc=$rc aggregate=${agg:-none} TFLOPS"
if [[ "$rc" -ne 0 || -z "$agg" || "$agg" == "0.00" ]]; then
  say "ABORT: smoke test produced no TFLOPS — not committing to the full sweep."
  say "       see $DRV/smoke.log and $SMOKE/"
  exit 1
fi
say "smoke OK"

# ---- stage 2: full gst sweep (~40-60 min) --------------------------------------
say "STAGE 2/4 gst sweep: N=1..8 x 9 precisions x 30s"
SWEEP=$LOG_ROOT/rvs/sweep_$TS
GPU_COUNTS="1 2 3 4 5 6 7 8" DURATION_MS=30000 OUT_DIR="$SWEEP" \
  ./run_tflops.sh >"$DRV/sweep.log" 2>&1
say "sweep rc=$? -> $SWEEP"
say "sweep rows: $(( $(wc -l <"$SWEEP/summary.csv" 2>/dev/null || echo 1) - 1 ))"

# ---- stage 3: health modules (20 min - 2.5 h incl. level configs) --------------
say "STAGE 3/4 health modules"
./run_rvs_health.sh >"$DRV/health.log" 2>&1
say "health rc=$?"
HEALTH=$(ls -dt "$LOG_ROOT"/rvs/health_* 2>/dev/null | head -1)
say "health dir: ${HEALTH:-none}"

# ---- stage 4: analysis ---------------------------------------------------------
say "STAGE 4/4 analysis"
$PY analyze_rvs.py "$SWEEP" -o "$BENCH_ROOT/results" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/rvs_tflops.{md,csv}"

{ echo; echo "SWEEP_DIR=$SWEEP"; echo "HEALTH_DIR=${HEALTH:-}"; echo "SMOKE_DIR=$SMOKE"; } >>"$STATE"
say "PART A DONE"
