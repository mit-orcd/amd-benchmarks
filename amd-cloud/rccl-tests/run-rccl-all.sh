#!/usr/bin/env bash
# All-collective RCCL sweep, N=2..8. Reproduces dell-cloud summary-rccl.md §1.1.
# RCCL collectives ONLY -- nothing Megatron-related belongs in this directory.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
BIN="${RCCL_TESTS_DIR:-$BENCH_ROOT/rccl-tests/src/build}"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/rccl_${OUT_TAG:-all}_$TS; mkdir -p "$OUT"
SUM=$OUT/rccl_summary.txt

GPU_COUNTS="${GPU_COUNTS:-2 3 4 5 6 7 8}"
COLLECTIVES="${COLLECTIVES:-all_reduce all_gather reduce_scatter broadcast reduce gather scatter alltoall alltoallv sendrecv}"
MIN_BYTES="${MIN_BYTES:-16M}"; MAX_BYTES="${MAX_BYTES:-8G}"; STEP_FACTOR="${STEP_FACTOR:-2}"
ITERS="${ITERS:-20}"; WARMUP="${WARMUP:-5}"

while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done < <(rccl_env)
assert_gpus_idle

{ echo "RCCL all-collective sweep $TS"; echo "bins: $BIN";
  echo "env : $(rccl_env | tr '\n' ' ')"; echo;
  printf '%-14s %3s %12s %14s\n' collective N max_size busbw_GB/s; } | tee "$SUM"

for coll in $COLLECTIVES; do
  exe="$BIN/${coll}_perf"
  [[ -x "$exe" ]] || { echo "SKIP $coll (no binary)" | tee -a "$SUM"; continue; }
  # alltoall/alltoallv allocate N x buffers per rank; the reference run OOMed at N=5.
  # Cap them so the sweep completes instead of dying mid-way.
  maxb="$MAX_BYTES"
  [[ "$coll" == alltoallv || "$coll" == alltoall ]] && maxb="${ALLTOALL_MAX:-4G}"
  for N in $GPU_COUNTS; do
    log="$OUT/${coll}_n${N}.log"
    timeout 1200 "$exe" -b "$MIN_BYTES" -e "$maxb" -f "$STEP_FACTOR" \
        -g "$N" -n "$ITERS" -w "$WARMUP" -c 1 >"$log" 2>&1
    rc=$?
    # last data row: in-place busbw is the second-to-last column
    read -r size busbw < <(awk '/^ *[0-9]/ {sz=$1; bw=$(NF-1)} END {print sz, bw}' "$log")
    printf '%-14s %3s %12s %14s   rc=%s\n' "$coll" "$N" "${size:--}" "${busbw:--}" "$rc" | tee -a "$SUM"
  done
done
echo "results: $OUT" | tee -a "$SUM"
