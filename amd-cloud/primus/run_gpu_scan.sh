#!/usr/bin/env bash
# Quick GEMM-only scan 1..8 GPUs (~5 min) -- smoke test / gate before the full sweep.
#
# Deviation from plan.md: we do NOT bind-mount the host Primus clone over the image's
# /workspace. The host clone (HEAD abc46648) has drifted from the image's pinned tree
# (b511d1b6), and shadowing it is exactly the API-drift failure the dell-cloud notes
# warn about. We run the image's own /workspace/Primus and bind-mount only an output
# directory at /out.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/gpu-scan-$RUN_ID; mkdir -p "$BASE"
OUT_HOST=$BENCH_ROOT/primus/sweep_out_$RUN_ID; mkdir -p "$OUT_HOST"
SUM=$BASE/summary.txt
assert_gpus_idle

{ echo "Primus GPU scan $RUN_ID"; echo "Image     : $IMG";
  echo "Driver log: $BASE"; echo "Bench out : $OUT_HOST";
  echo "Started   : $(date -Iseconds)"; echo; } | tee "$SUM"

for N in 1 2 3 4 5 6 7 8; do
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  log=$BASE/scan_${N}gpu.log; start=$(date +%s)
  echo "----- gemm N=$N devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout 600 docker run $(dgpu_args) \
      -v "$OUT_HOST":/out -w /workspace/Primus \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      "$IMG" bash -c \
      "./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10 \
         --output-file /out/gemm_N${N}.md" >"$log" 2>&1
  rc=$?
  echo "  rc=$rc duration=$(($(date +%s)-start))s log=$log" | tee -a "$SUM"
done

echo "$RUN_ID" > "$LOG_ROOT/primus/CURRENT_RUN_ID.txt"
echo "Finished  : $(date -Iseconds)" | tee -a "$SUM"
