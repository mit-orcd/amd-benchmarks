#!/usr/bin/env bash
# 5 RCCL configs x {all_reduce, all_gather} x N=2..8 -- isolates the N=5/6/7 cliff.
# This is what tells you *which knob* recovers a cliff, if any does.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
BIN="${RCCL_TESTS_DIR:-$BENCH_ROOT/rccl-tests/src/build}"
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rccl/rccl_tests_$TS; mkdir -p "$OUT"
SUM=$OUT/rccl_tests_summary.txt
GPU_COUNTS="${GPU_COUNTS:-2 3 4 5 6 7 8}"
COLLECTIVES="${COLLECTIVES:-all_reduce all_gather}"
CONFIGS="${CONFIGS:-default tree ring no_mscll proto_simple}"

apply_config() {
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=0 \
         NCCL_SHM_DISABLE=0 NCCL_DEBUG=WARN
  export RCCL_MSCCL_ENABLE=1 NCCL_ALGO=Ring,Tree NCCL_PROTO=Simple,LL,LL128
  case "$1" in
    default)      ;;
    tree)         export NCCL_ALGO=Tree ;;
    ring)         export NCCL_ALGO=Ring ;;
    no_mscll)     export RCCL_MSCCL_ENABLE=0 ;;
    proto_simple) export NCCL_PROTO=Simple ;;
  esac
}

{ echo "RCCL config sweep $TS"; echo "bins: $BIN"; echo;
  printf '%-12s %-14s %3s %12s %14s\n' config collective N max_size busbw_GB/s; } | tee "$SUM"
assert_gpus_idle

for cfg in $CONFIGS; do
  apply_config "$cfg"
  for coll in $COLLECTIVES; do
    exe="$BIN/${coll}_perf"; [[ -x "$exe" ]] || continue
    for N in $GPU_COUNTS; do
      log="$OUT/${coll}_${cfg}_n${N}.log"
      timeout 900 "$exe" -b 16M -e 8G -f 2 -g "$N" -n 20 -w 5 -c 1 >"$log" 2>&1
      read -r size busbw < <(awk '/^ *[0-9]/ {sz=$1; bw=$(NF-1)} END {print sz, bw}' "$log")
      printf '%-12s %-14s %3s %12s %14s\n' "$cfg" "$coll" "$N" "${size:--}" "${busbw:--}" | tee -a "$SUM"
    done
  done
done
echo "results: $OUT" | tee -a "$SUM"
