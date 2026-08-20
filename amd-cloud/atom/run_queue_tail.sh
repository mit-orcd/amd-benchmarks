#!/usr/bin/env bash
# Final tail queue. Kimi-K3 first, then everything else.
#
# Waits for run_queue_34_retry.sh (#3 ISL=4096 done, #4 repeats config A in progress), then:
#
#   KIMI-K3 (priority)
#     T1  matched-cap EP A/B     -> kimi-k3-ep-matched.md    ~1.7 h
#                                   Closes the last open Kimi question. Also re-pulls the
#                                   MAD image that K5 deleted, which T2 then needs.
#     T2  repeats config B (MAD) -> kimi-k3-repeats.md       ~40 min
#                                   #4 could only run config A: K5 had deleted the MAD image,
#                                   so run_repeats.sh hit "SKIP B_mad: image absent" and the
#                                   "is the 9% MAD gap real or noise" question went unanswered.
#                                   T1 restores the image; this runs the missing arm and
#                                   re-folds A+B into one analysis.
#   NON-KIMI
#     N1  megatron-ref rerun     -> PRIMUS_REPORT.md 1.2a    ~15 min
#                                   Corrected flag set (swiglu/RMSNorm/rope/disable-bias/
#                                   untie/overlap-*/log-throughput) -- see that script's header.
#
# Every stage is independent; a failure never blocks the next. Each waits for idle GPUs.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_tail_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

wait_for_gpus() {
  sleep 30
  for i in $(seq 1 80); do
    local busy
    busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
    [[ "${busy:-0}" -eq 0 ]] && return 0
    (( i % 10 == 0 )) && say "  waiting for GPUs ($busy busy)"
    sleep 30
  done
  say "  WARNING: GPUs still busy after 40 min, proceeding"
}

say "tail queue start — Kimi-K3 stages first, then megatron-ref"

if pgrep -f 'run_queue_34_retry\.sh' >/dev/null 2>&1; then
  say "waiting for run_queue_34_retry.sh (#4 repeats config A)..."
  waited=0
  while pgrep -f 'run_queue_34_retry\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting (${waited}s)"
    if (( waited > 21600 )); then say "ABORT: prior queue >6 h"; exit 1; fi
  done
  say "prior queue finished after ${waited}s"
else
  say "prior queue not running — proceeding"
fi
wait_for_gpus

# ---------------- T1: matched-cap EP A/B ----------------
say "===== T1_ep_matched ====="
./run_ep_matched.sh >"$DRV/T1_ep_matched.log" 2>&1
rc=$?; say "T1_ep_matched rc=$rc"
[[ $rc -ne 0 ]] && tail -15 "$DRV/T1_ep_matched.log" | tee -a "$STATE"
echo "T1_RC=$rc" >>"$STATE"
wait_for_gpus

# ---------------- T2: repeats config B (MAD image) ----------------
# Skip config A by pointing IMG_A at a tag that does not exist -- run_repeats.sh then logs
# "SKIP A_original: image absent" and runs only B. Afterwards the A results from #4 are
# copied in so the analyzer sees both arms and produces the real A-vs-B comparison.
say "===== T2_repeats_mad ====="
PRIOR_REP=$(ls -dt "$LOG_ROOT"/atom/kimi_repeats_* 2>/dev/null | head -1)
say "prior repeats dir (config A): ${PRIOR_REP:-none}"
IMG_A="rocm/atom-dev:__skip_config_a__" PORT=8015 ./run_repeats.sh >"$DRV/T2_repeats_mad.log" 2>&1
rc=$?; say "T2_repeats_mad rc=$rc"
[[ $rc -ne 0 ]] && tail -15 "$DRV/T2_repeats_mad.log" | tee -a "$STATE"
echo "T2_RC=$rc" >>"$STATE"

NEW_REP=$(ls -dt "$LOG_ROOT"/atom/kimi_repeats_* 2>/dev/null | head -1)
if [[ -n "$PRIOR_REP" && -n "$NEW_REP" && "$PRIOR_REP" != "$NEW_REP" ]]; then
  nA=$(ls "$PRIOR_REP"/A_original_rep*.json 2>/dev/null | wc -l)
  nB=$(ls "$NEW_REP"/B_mad_rep*.json 2>/dev/null | wc -l)
  say "merging config A ($nA files, from $(basename "$PRIOR_REP")) into $(basename "$NEW_REP") ($nB B files)"
  if [[ "$nA" -gt 0 && "$nB" -gt 0 ]]; then
    cp "$PRIOR_REP"/A_original_rep*.json "$NEW_REP"/ 2>/dev/null
    cp "$PRIOR_REP"/A_original_rep*.log  "$NEW_REP"/ 2>/dev/null
    echo "MERGED_A_FROM=$PRIOR_REP" >>"$NEW_REP/STATE.txt"
    say "re-running analyze_repeats.py on the merged dir"
    $PY analyze_repeats.py "$NEW_REP" -o "$BENCH_ROOT/results" >"$DRV/T2_merge_analyze.log" 2>&1
    say "merged analyze rc=$? -> results/kimi-k3-repeats.md"
  else
    say "not merging: nA=$nA nB=$nB — leaving the single-arm analysis as written"
  fi
fi
wait_for_gpus

say "### ALL KIMI-K3 WORK COMPLETE ###"

# ---------------- N1: megatron-ref rerun ----------------
say "===== N1_megatron_ref ====="
( cd ../megatron-ref && ./run_megatron_ref.sh 8 ) >"$DRV/N1_megatron_ref.log" 2>&1
rc=$?; say "N1_megatron_ref rc=$rc"
tail -20 "$DRV/N1_megatron_ref.log" | tee -a "$STATE"
echo "N1_RC=$rc" >>"$STATE"

say "TAIL QUEUE DONE — all remaining GPU work complete"
grep '_RC=' "$STATE" | tee -a "$STATE"
