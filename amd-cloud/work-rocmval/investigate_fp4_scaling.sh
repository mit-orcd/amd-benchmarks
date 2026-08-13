#!/usr/bin/env bash
# Investigate the fp4 GPU-count scaling anomaly found in the first gst sweep
# (results/rvs_tflops.md, N=8 per-GPU spread up to 63% vs <1% for every other precision,
# including fp6/bf6 at the same 10,000 TFLOPS peak class and the same MX block-scaling
# mechanism). That pattern rules out ordinary shared power/thermal budget as the sole
# explanation -- power capping depresses every die roughly equally, not one die at ~99% of
# solo peak next to another at <40%.
#
# This script does NOT explain the anomaly by itself. It gathers the two things the first
# sweep did not capture, so a human (or a follow-up analysis pass) can actually distinguish
# the competing hypotheses:
#   1. Per-GPU clock/power telemetry sampled DURING each run, at 2 s resolution, via
#      `rocm-smi --showclocks --showpower`. If low performers show depressed clocks
#      alongside their low GFLOPS, that is evidence for a real thermal/power/silicon cause.
#      If clocks look normal while GFLOPS is low, that points at a measurement or
#      launch/scheduling artifact in RVS's parallel gst path instead.
#   2. THREE repeats at each of N=5,6,7,8 (fp4 only; N=1..4 were clean in the first sweep
#      and are not repeated here). If the same GPU ID is consistently the low performer
#      across repeats, that is a deterministic, hardware- or topology-correlated effect.
#      If the low performer changes between repeats, that points at non-determinism in the
#      launch/sync path -- a software bug, not a die.
#
# Usage: ./investigate_fp4_scaling.sh
# Runtime: ~15-20 min (4 GPU-counts x 3 repeats x ~30 s gst duration, plus RVS overhead).
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rvs/fp4_investigation_$TS; mkdir -p "$OUT"
SUM=$OUT/investigation_summary.txt
REPEATS="${REPEATS:-3}"
DURATION_MS="${DURATION_MS:-30000}"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  echo "ABORT: $busy GPU(s) busy — another benchmark is running." >&2
  exit 1
fi

{ echo "fp4 scaling investigation $TS"; echo "repeats=$REPEATS duration_ms=$DURATION_MS";
  echo; } | tee "$SUM"

for N in 5 6 7 8; do
  for r in $(seq 1 "$REPEATS"); do
    tag="n${N}_r${r}"
    run_dir="$OUT/$tag"; mkdir -p "$run_dir"
    clocklog="$run_dir/clocks_power.log"
    echo "----- N=$N repeat=$r -----" | tee -a "$SUM"

    # Background sampler: per-GPU clock (SCLK) and power draw every 2 s, timestamped.
    ( while true; do
        echo "=== $(date +%s.%N) ==="
        rocm-smi --showclocks --showpower 2>/dev/null
        sleep 2
      done ) >"$clocklog" 2>&1 &
    sampler_pid=$!

    GPU_COUNTS="$N" PRECISIONS=fp4 DURATION_MS="$DURATION_MS" OUT_DIR="$run_dir" \
      ./run_tflops.sh >"$run_dir/gst.log" 2>&1
    rc=$?

    kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null

    # Per-GPU peaks for this repeat, sorted, so eyeballing which GPU is low is immediate.
    peaks=$(grep -oE 'gpu[0-9]+=[0-9.]+' "$run_dir/summary.csv" 2>/dev/null | tr ';' '\n' | sort -t= -k2 -n)
    echo "  rc=$rc" | tee -a "$SUM"
    echo "$peaks" | sed 's/^/    /' | tee -a "$SUM"
  done
done

echo | tee -a "$SUM"
echo "results: $OUT" | tee -a "$SUM"
echo "Next: $PY analyze_fp4_scaling.py $OUT -o \$BENCH_ROOT/results"
