#!/usr/bin/env bash
# Final queue: waits for run_queue_3_4.sh (which itself waits for #1), then runs the
# remaining outstanding work before this server goes away.
#
#   1. megatron-ref  -- the apples-to-apples GPT-15.6B run vs B200 / Dell MI355X.
#                       Has NEVER completed: failed twice (hipBLASLt algo selection, then
#                       RCCL HSA_NO_SCRATCH_RECLAIM). Both fixes are in the script but it
#                       was never relaunched, so PRIMUS_REPORT.md section 1.2a is missing.
#                       Highest-value unfinished item outside Part D.
#   2. bandwidth health -- Part A's pebb/pbqt/babel were measured while the 1.5 TB Kimi
#                       download saturated NVMe (pebb showed NUMA page-allocation errors).
#                       ~15 min on an idle box, and it is what licenses Part B's
#                       "algorithm, not fabric" conclusion.
#   3. profiling (#2) -- replaces the residual-based step-time estimate with measurement.
#
# Ordered by value-if-time-runs-out. Each stage is independent: a failure does not block
# the next. Every stage self-checks the GPUs are free first.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_final_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

wait_for_gpus() {
  sleep 30
  for i in $(seq 1 80); do
    local busy
    busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
    [[ "${busy:-0}" -eq 0 ]] && return 0
    (( i % 10 == 0 )) && say "  waiting for GPUs to free ($busy busy)"
    sleep 30
  done
  say "  WARNING: GPUs still busy after 40 min, proceeding anyway"
  return 0
}

say "final queue start — megatron-ref, bandwidth health, profiling"

# ---- wait for the earlier queue (and thus #1) ------------------------------------
if pgrep -f 'run_queue_3_4\.sh' >/dev/null 2>&1; then
  say "waiting for run_queue_3_4.sh to finish..."
  waited=0
  while pgrep -f 'run_queue_3_4\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting (${waited}s)"
    if (( waited > 43200 )); then say "ABORT: prior queue still running after 12 h"; exit 1; fi
  done
  say "prior queue finished after ${waited}s"
else
  say "prior queue not running — proceeding"
fi
wait_for_gpus

# ---- 1. megatron-ref (highest value) ---------------------------------------------
say "===== STAGE 1/3: megatron-ref (GPT-15.6B, vs B200 / Dell) ====="
( cd ../megatron-ref && ./run_megatron_ref.sh 8 ) >"$DRV/megatron_ref.log" 2>&1
rc1=$?
say "megatron-ref rc=$rc1"
if [[ $rc1 -ne 0 ]]; then
  say "megatron-ref FAILED — last 20 lines:"; tail -20 "$DRV/megatron_ref.log" | tee -a "$STATE"
else
  say "megatron-ref OK -> PRIMUS_REPORT.md section 1.2a should now exist"
fi
wait_for_gpus

# ---- 2. bandwidth health recheck -------------------------------------------------
say "===== STAGE 2/3: RVS bandwidth health (clean pebb/pbqt/babel) ====="
( cd ../work-rocmval && ./rerun_bandwidth_health.sh ) >"$DRV/bandwidth.log" 2>&1
rc2=$?
say "bandwidth health rc=$rc2 (results in logs/rvs/health_bw_*)"
[[ $rc2 -ne 0 ]] && tail -12 "$DRV/bandwidth.log" | tee -a "$STATE"
wait_for_gpus

# ---- 3. profiling ----------------------------------------------------------------
say "===== STAGE 3/3: profiler trace capture ====="
./run_profile.sh >"$DRV/profile.log" 2>&1
rc3=$?
say "profiling rc=$rc3"
[[ $rc3 -ne 0 ]] && tail -15 "$DRV/profile.log" | tee -a "$STATE"

{ echo; echo "MEGATRON_REF_RC=$rc1"; echo "BANDWIDTH_RC=$rc2"; echo "PROFILE_RC=$rc3"; } >>"$STATE"
say "FINAL QUEUE DONE (megatron-ref=$rc1, bandwidth=$rc2, profile=$rc3)"
say "all GPU work for this server is now complete"
