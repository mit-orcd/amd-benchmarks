#!/usr/bin/env bash
# Full 1..8 GPU x 9 precision RVS gst sweep (~40 min at DURATION_MS=30000).
set -euo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$LOG_ROOT/rvs/sweep_$TS"; mkdir -p "$OUT"
assert_gpus_idle
GPU_COUNTS="${GPU_COUNTS:-1 2 3 4 5 6 7 8}" \
DURATION_MS="${DURATION_MS:-30000}" \
OUT_DIR="$OUT" ./run_tflops.sh 2>&1 | tee "$OUT/sweep.log"
echo "results: $OUT"
