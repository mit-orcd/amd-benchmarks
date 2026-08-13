#!/usr/bin/env bash
# Chains §A.7: investigate -> analyze -> fold the verdict into results/rvs_tflops.md.
# Designed to run under nohup/setsid and survive logout, like run_part_a.sh / run_part_b.sh.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

DRV=$LOG_ROOT/rvs/fp4_investigation_driver_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "fp4 investigation driver start"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy."
  exit 1
fi
say "GPUs idle, proceeding"

say "STAGE 1/3 investigate_fp4_scaling.sh"
./investigate_fp4_scaling.sh >"$DRV/investigate.log" 2>&1
rc=$?
inv_dir=$(ls -dt "$LOG_ROOT"/rvs/fp4_investigation_* 2>/dev/null | grep -v _driver_ | head -1)
say "investigate rc=$rc -> $inv_dir"
if [[ $rc -ne 0 || -z "$inv_dir" ]]; then
  say "ABORT: investigation run failed, not proceeding to analysis. See $DRV/investigate.log"
  exit 1
fi

say "STAGE 2/3 analyze_fp4_scaling.py"
$PY analyze_fp4_scaling.py "$inv_dir" -o "$BENCH_ROOT/results" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/fp4_investigation.md"

say "STAGE 3/3 fold result into rvs_tflops.md"
$PY update_rvs_summary_with_investigation.py \
  "$BENCH_ROOT/results/fp4_investigation.md" \
  "$BENCH_ROOT/results/rvs_tflops.md" >"$DRV/update.log" 2>&1
say "update rc=$? -> $BENCH_ROOT/results/rvs_tflops.md"

say "DONE"
