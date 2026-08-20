#!/usr/bin/env bash
# Retry driver for #3/#4: the original run_queue_3_4.sh's wait-for-#1 pgrep loop never
# saw #1 exit (it had genuinely finished at 2026-08-19T22:24, log-confirmed) and aborted
# after a 6h timeout, so #3/#4 never ran. #1 is long done; GPUs are idle now. Runs #3
# then #4 directly, then folds both into kimi-k3-improve.md -- identical tail logic to
# the original script, just without the broken wait.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_34_retry_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "retry queue start -- #1 already done, running #3 then #4 directly"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy"; exit 1; }

say "===== STAGE #3: ISL=4096 sweep ====="
./run_isl4096.sh >"$DRV/isl4096.log" 2>&1
rc3=$?
say "#3 rc=$rc3"
[[ $rc3 -ne 0 ]] && { say "#3 failed"; tail -12 "$DRV/isl4096.log" | tee -a "$STATE"; }

sleep 30
for i in $(seq 1 60); do
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
  [[ "${busy:-0}" -eq 0 ]] && break
  sleep 30
done

say "===== STAGE #4: repeats at c=64 ====="
./run_repeats.sh >"$DRV/repeats.log" 2>&1
rc4=$?
say "#4 rc=$rc4"
[[ $rc4 -ne 0 ]] && { say "#4 failed"; tail -12 "$DRV/repeats.log" | tee -a "$STATE"; }

say "===== updating results/kimi-k3-improve.md ====="
ISL_DIR=$(ls -dt "$LOG_ROOT"/atom/kimi_isl4096_* 2>/dev/null | head -1)
REP_DIR=$(ls -dt "$LOG_ROOT"/atom/kimi_repeats_* 2>/dev/null | head -1)
$PY update_improve_with_3_4.py \
   --isl "${ISL_DIR:-none}" --repeats "${REP_DIR:-none}" \
   "$BENCH_ROOT/results/kimi-k3-improve.md" >"$DRV/update.log" 2>&1
say "update rc=$? -> results/kimi-k3-improve.md"

echo "RC3=$rc3" >>"$STATE"; echo "RC4=$rc4" >>"$STATE"
say "RETRY QUEUE DONE (#3 rc=$rc3, #4 rc=$rc4)"
