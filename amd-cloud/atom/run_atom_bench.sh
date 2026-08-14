#!/usr/bin/env bash
# Drive a concurrency sweep against a already-running ATOM server.
#
# Usage: ./run_atom_bench.sh <model_dir> [PORT] [ISL] [OSL] [CONC_LIST]
#   ./run_atom_bench.sh $SCRATCH/models/Qwen3-8B-FP8 8000 1024 1024 "1 4 16 64"
#
# Calls atom.benchmarks.benchmark_serving directly rather than ATOM's
# scripts/run_benchmark.sh, because that script hardcodes OUTPUT_DIR=/app/logs_claude and
# overwrites a single benchmark.log per run. We write one JSON + one log per concurrency
# into the tracked log tree instead.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh

MODEL="${1:?usage: $0 <model_dir> [PORT] [ISL] [OSL] [CONC_LIST]}"
PORT="${2:-$(cat "$LOG_ROOT/atom/CURRENT_PORT.txt" 2>/dev/null || echo 8000)}"
ISL="${3:-1024}"
OSL="${4:-1024}"
CONC_LIST="${5:-1 2 4 8 16 32 64 128 256}"

IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-10}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/sweep_${TS}; mkdir -p "$OUT"
SUM=$OUT/summary.txt

curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || {
  echo "ERROR: no ATOM server responding on port $PORT. Start it with ./run_atom_server.sh" >&2
  exit 1
}

{ echo "ATOM serving sweep $TS"; echo "model   : $MODEL"; echo "port    : $PORT";
  echo "ISL/OSL : $ISL/$OSL"; echo "conc    : $CONC_LIST"; echo "image   : $IMG";
  echo "started : $(date -Iseconds)"; echo;
  printf '%6s %12s %12s %12s %12s\n' conc req/s out_tok/s ttft_ms_med tpot_ms_med; } | tee "$SUM"

for C in $CONC_LIST; do
  n=$(( C * PROMPT_MULTIPLIER ))
  log="$OUT/c${C}.log"; json="$OUT/c${C}.json"
  start=$(date +%s)
  timeout "${BENCH_TIMEOUT:-1800}" docker run --rm --network host \
    -v "$MODEL":/model:ro -v "$OUT":/out \
    "$IMG" bash -c "
      python -m atom.benchmarks.benchmark_serving \
        --model=/model --backend=vllm --base-url=http://localhost:${PORT} \
        --dataset-name=random \
        --random-input-len=$ISL --random-output-len=$OSL \
        --random-range-ratio=$RANDOM_RANGE_RATIO \
        --max-concurrency=$C --num-prompts=$n \
        --num-warmups=$(( C * 2 )) \
        --request-rate=inf --ignore-eos --trust-remote-code \
        --save-result --result-filename=/out/c${C}.json \
        --percentile-metrics=ttft,tpot,itl,e2el
    " >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  if [[ -f "$json" ]]; then
    read -r rps ots ttft tpot completed < <($PY - "$json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("request_throughput", 0), d.get("output_throughput", 0),
      d.get("median_ttft_ms", 0), d.get("median_tpot_ms", 0), d.get("completed", 0))
PY
)
    printf '%6s %12.2f %12.1f %12.1f %12.2f   rc=%s %ss completed=%s\n' \
      "$C" "$rps" "$ots" "$ttft" "$tpot" "$rc" "$dur" "$completed" | tee -a "$SUM"
    # benchmark_serving exits 0 even when EVERY request fails -- it just warns and writes
    # zeros. Without this check a totally dead server yields a full sweep of 0.00 rows and
    # a cheerful rc=0. Treat "no request completed" as fatal and stop immediately rather
    # than burning the remaining concurrency points and the later tiers.
    if [[ "${completed:-0}" -eq 0 ]]; then
      echo "FATAL: 0 requests completed at concurrency $C — the server is rejecting requests." | tee -a "$SUM"
      echo "       Check the server log for HTTP 4xx/5xx (a model-id mismatch returns 400" | tee -a "$SUM"
      echo "       while /v1/models still answers 200). Aborting sweep." | tee -a "$SUM"
      echo "       bench log: $log" | tee -a "$SUM"
      exit 1
    fi
  else
    printf '%6s %12s %12s %12s %12s   rc=%s %ss (no json)\n' \
      "$C" - - - - "$rc" "$dur" | tee -a "$SUM"
  fi
done

echo "results: $OUT" | tee -a "$SUM"
echo "$OUT" > "$LOG_ROOT/atom/CURRENT_SWEEP_DIR.txt"
