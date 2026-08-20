#!/usr/bin/env bash
# Master tail queue — replaces run_queue_final.sh, which had non-Kimi work ahead of Kimi
# work. Kimi-K3 is the priority test, so ALL Kimi stages run before anything else.
#
# Waits for run_queue_3_4.sh (which itself waits for #1 max-num-seqs 512), then:
#
#   KIMI-K3 (priority)
#     K1  profiling (#2)        -> kimi-k3-profile.md       ~1 h
#     K2  EP retest on MAD img  -> logs only, ~10 min if it fails at load
#     K3  max-num-seqs 1024     -> kimi-k3-maxseqs1024.md   ~2-3 h
#     K4  max-num-seqs 2048     -> kimi-k3-maxseqs2048.md   ~3 h
#                                  (#1 hit 3,386 tok/s at c=512 still climbing -- no knee yet)
#     K5  EP on fresh :latest   -> logs, ~15 min + pull
#                                  MUST be last: re-pulls :latest, which K1/K3/K4 use.
#   NON-KIMI (after all Kimi)
#     N1  megatron-ref          -> PRIMUS_REPORT.md 1.2a   ~1 h
#     N2  bandwidth health      -> logs/rvs/health_bw_*    ~15 min
#
# Ordered so that if the server disappears early, the most valuable Kimi work is already
# done. Every stage is independent — a failure never blocks the next. Each waits for the
# GPUs to actually be free first.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/queue_all_$TS; mkdir -p "$DRV"
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

stage() {  # stage <label> <dir> <cmd...>
  local label=$1 dir=$2; shift 2
  say "===== $label ====="
  ( cd "$dir" && "$@" ) >"$DRV/${label}.log" 2>&1
  local rc=$?
  say "$label rc=$rc"
  [[ $rc -ne 0 ]] && tail -12 "$DRV/${label}.log" | tee -a "$STATE"
  echo "${label}_RC=$rc" >>"$STATE"
  wait_for_gpus
  return 0
}

say "master queue start — ALL Kimi-K3 stages first, then non-Kimi"

if pgrep -f 'run_queue_3_4\.sh' >/dev/null 2>&1; then
  say "waiting for run_queue_3_4.sh (holds #1, #3, #4)..."
  waited=0
  while pgrep -f 'run_queue_3_4\.sh' >/dev/null 2>&1; do
    sleep 60; waited=$((waited+60))
    (( waited % 1800 == 0 )) && say "  still waiting (${waited}s)"
    if (( waited > 50400 )); then say "ABORT: prior queue >14 h"; exit 1; fi
  done
  say "prior queue finished after ${waited}s"
else
  say "prior queue not running — proceeding"
fi
wait_for_gpus

# ---------------- KIMI-K3 (priority) ----------------
stage K1_profiling   .             ./run_profile.sh
stage K2_ep_mad      .             ./run_ep_mad.sh
stage K3_maxseqs1024 .             ./run_kimi_1024.sh
stage K4_maxseqs2048 .             ./run_kimi_2048.sh
# K5 MUST be last among Kimi stages: it re-pulls :latest, which K1/K3/K4 depend on.
stage K5_ep_fresh    .             ./run_ep_fresh_latest.sh

say "### ALL KIMI-K3 WORK COMPLETE ###"

# ---------------- NON-KIMI ----------------
stage N1_megatron_ref ../megatron-ref ./run_megatron_ref.sh 8
stage N2_bandwidth    ../work-rocmval ./rerun_bandwidth_health.sh

say "MASTER QUEUE DONE — all GPU work for this server is complete"
grep '_RC=' "$STATE" | tee -a "$STATE"
