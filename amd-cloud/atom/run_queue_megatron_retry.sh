#!/usr/bin/env bash
# Waits for the #3/#4 retry queue, then reruns megatron-ref with the corrected flag set
# (Dell-matched: swiglu/RMSNorm/rope/disable-bias/untie/overlap-*/log-throughput).
# The 2026-08-20 07:45 run was not apples-to-apples -- see run_megatron_ref.sh header.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_megatron_retry_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "megatron retry queue start"

if pgrep -f 'run_queue_34_retry\.sh' >/dev/null 2>&1; then
  say "waiting for run_queue_34_retry.sh..."
  waited=0
  while pgrep -f 'run_queue_34_retry\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting (${waited}s)"
    if (( waited > 43200 )); then say "ABORT: prior queue >12 h"; exit 1; fi
  done
  say "prior queue finished after ${waited}s"
fi

sleep 60
for i in $(seq 1 80); do
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
  [[ "${busy:-0}" -eq 0 ]] && break
  sleep 30
done

say "===== megatron-ref rerun (corrected flags) ====="
( cd ../megatron-ref && ./run_megatron_ref.sh 8 ) >"$DRV/megatron.log" 2>&1
rc=$?
say "megatron-ref rc=$rc"
tail -20 "$DRV/megatron.log" | tee -a "$STATE"
echo "MEGATRON_RC=$rc" >>"$STATE"
say "MEGATRON RETRY DONE"
