#!/usr/bin/env bash
# Waits for the single-stream Kimi experiment, then bisects the megatron-ref SIGSEGV.
#
# History: the 07:45 run completed but was NOT Dell-comparable (8 flags missing, ~3x slower
# on a cheaper model). Adding those 8 flags fixed comparability -- the model now builds with
# 16,223,016,960 params, matching Dell exactly -- but it then SIGSEGV'd on the FIRST training
# step, after RCCL init. So the breakage is in the newly-added flags, not the model shape.
#
# Bisect order, most-likely culprit first. Each attempt is ~4 min to crash, ~8 min to succeed.
#   A) OVERLAP=0        drops --overlap-grad-reduce/--overlap-param-gather. These are pure DP
#                       comm optimizations: the model stays byte-for-byte Dell-comparable and
#                       only the overlap differs, so a result here is still usable against
#                       Dell's 790.4 (with a stated caveat that overlap was off).
#   B) OVERLAP=0 GAF=0  additionally drops gradient-accumulation-fusion.
# Stops at the first attempt that produces a parsed TFLOP/s figure.
#
# Not retried: the pre-correction flag set. That run already succeeded and its number is
# unusable for the Dell comparison, so repeating it would burn GPU time for nothing.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_megatron_bisect_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "megatron bisect queue start"

if pgrep -f 'run_single_stream\.sh' >/dev/null 2>&1; then
  say "waiting for run_single_stream.sh (Kimi priority)..."
  waited=0
  while pgrep -f 'run_single_stream\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting (${waited}s)"
    if (( waited > 28800 )); then say "ABORT: single-stream >8 h"; exit 1; fi
  done
  say "single-stream finished after ${waited}s"
else
  say "single-stream not running — proceeding"
fi

wait_gpus() {
  sleep 60
  for i in $(seq 1 80); do
    local busy
    busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
    [[ "${busy:-0}" -eq 0 ]] && return 0
    sleep 30
  done
  say "  WARNING: GPUs still busy, proceeding"
}

attempt() {  # attempt <label> <env assignments...>
  local label=$1; shift
  wait_gpus
  say "===== megatron attempt $label ($*) ====="
  ( cd ../megatron-ref && env "$@" ./run_megatron_ref.sh 8 ) >"$DRV/${label}.log" 2>&1
  local rc=$?
  local tf
  tf=$(grep -oE 'TFLOPS=[0-9.]+' "$(ls -dt "$LOG_ROOT"/megatron-ref/run_* | head -1)/STATE.txt" 2>/dev/null | tail -1 | cut -d= -f2)
  say "$label rc=$rc parsed_TFLOPS=${tf:-none}"
  echo "${label}_RC=$rc" >>"$STATE"; echo "${label}_TFLOPS=${tf:-}" >>"$STATE"
  if [[ -n "$tf" ]]; then
    say "SUCCESS on $label — TF/s/GPU = $tf (Dell reference: 790.4, B200: 986.0)"
    return 0
  fi
  tail -12 "$DRV/${label}.log" | tee -a "$STATE"
  return 1
}

if attempt A_no_overlap OVERLAP=0 GAF=1; then
  say "MEGATRON BISECT DONE (A_no_overlap succeeded)"; exit 0
fi
if attempt B_no_overlap_no_gaf OVERLAP=0 GAF=0; then
  say "MEGATRON BISECT DONE (B_no_overlap_no_gaf succeeded)"; exit 0
fi

say "MEGATRON BISECT DONE — both attempts failed; megatron-ref remains unmeasured on this host"
say "PRIMUS_REPORT.md 1.2 therefore still carries external reference data only."
