#!/usr/bin/env bash
# Queue driver: wait for next-step #1 (run_kimi_512.sh) to finish, then run #3 and #4.
#
#   #3  ISL = 4096   (run_isl4096.sh)  -> results/kimi-k3-isl4096.md
#   #4  repeats c=64 (run_repeats.sh)  -> results/kimi-k3-repeats.md
#   then a single pass folding both into results/kimi-k3-improve.md
#
# Strictly sequential: all three need all 8 GPUs. Each stage runs only if the GPUs are
# actually free, and a failure in one does not block the next (they answer independent
# questions). Safe under setsid/nohup.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_3_4_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "queue start — will run #3 (ISL=4096) then #4 (repeats) after #1 completes"

# ---- wait for #1 -----------------------------------------------------------------
# Match the script path as invoked, and exclude our own pgrep, so this cannot match itself.
if pgrep -f 'run_kimi_512\.sh' >/dev/null 2>&1; then
  say "waiting for run_kimi_512.sh (#1) to finish..."
  waited=0
  while pgrep -f 'run_kimi_512\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting for #1 (${waited}s)"
    if (( waited > 21600 )); then say "ABORT: #1 still running after 6 h"; exit 1; fi
  done
  say "#1 finished after ${waited}s of waiting"
else
  say "#1 not running — proceeding immediately"
fi

# let VRAM drain and confirm the box is actually free
sleep 30
for i in $(seq 1 60); do
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
  [[ "${busy:-0}" -eq 0 ]] && break
  say "  GPUs still busy ($busy), waiting..."; sleep 30
done

# ---- #3: ISL = 4096 --------------------------------------------------------------
say "===== STAGE #3: ISL=4096 sweep ====="
./run_isl4096.sh >"$DRV/isl4096.log" 2>&1
rc3=$?
say "#3 rc=$rc3"
[[ $rc3 -ne 0 ]] && { say "#3 failed — see $DRV/isl4096.log"; tail -12 "$DRV/isl4096.log" | tee -a "$STATE"; }

sleep 30
for i in $(seq 1 60); do
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
  [[ "${busy:-0}" -eq 0 ]] && break
  sleep 30
done

# ---- #4: repeats -----------------------------------------------------------------
say "===== STAGE #4: repeats at c=64 ====="
./run_repeats.sh >"$DRV/repeats.log" 2>&1
rc4=$?
say "#4 rc=$rc4"
[[ $rc4 -ne 0 ]] && { say "#4 failed — see $DRV/repeats.log"; tail -12 "$DRV/repeats.log" | tee -a "$STATE"; }

# ---- fold both into kimi-k3-improve.md -------------------------------------------
say "===== updating results/kimi-k3-improve.md ====="
ISL_DIR=$(ls -dt "$LOG_ROOT"/atom/kimi_isl4096_* 2>/dev/null | head -1)
REP_DIR=$(ls -dt "$LOG_ROOT"/atom/kimi_repeats_* 2>/dev/null | head -1)
$PY update_improve_with_3_4.py \
   --isl "${ISL_DIR:-none}" --repeats "${REP_DIR:-none}" \
   "$BENCH_ROOT/results/kimi-k3-improve.md" >"$DRV/update.log" 2>&1
say "update rc=$? -> results/kimi-k3-improve.md"

echo "RC3=$rc3" >>"$STATE"; echo "RC4=$rc4" >>"$STATE"
say "QUEUE DONE (#3 rc=$rc3, #4 rc=$rc4)"
