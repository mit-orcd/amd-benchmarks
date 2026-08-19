#!/usr/bin/env bash
# Kimi-K3 rerun using AMD's official MAD benchmark recipe.
# Plan and rationale: ../notes-kimi-k3.md
#
# Differs from the original Part D tier-3 run in three ways, all taken from
# https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md :
#   1. pinned Kimi-K3-specific image instead of rocm/atom-dev:latest
#   2. 11 ATOM/AITER env vars that select the validated kernel set
#   3. --max-num-batched-tokens 10240 and NO --online_quant_config
#      (MAD relies on the env-var kernel selection instead)
# Sweep goes to 256 like MAD's official sweep, even though --max-num-seqs is 64;
# past 64 the server queues rather than rejects, which is the behaviour MAD characterizes.
#
# WRITES ONLY NEW FILES. Does not touch results/kimi-k3-base.md, results/atom.{md,csv}, or any
# existing logs/atom/sweep_* directory.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="${MAD_IMG:-rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
PORT="${PORT:-8010}"
CONC="${MAD_CONC:-64 128 256}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NAME="atom-kimi-mad"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_mad_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "Kimi-K3 MAD-recipe rerun start (out: $OUT)"
say "image=$IMG conc='$CONC' ISL/OSL=$ISL/$OSL"

# ---- guards -------------------------------------------------------------------
busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then say "ABORT: $busy GPU(s) busy."; exit 1; fi
if pgrep -af 'atom.entrypoints' >/dev/null 2>&1; then
  say "ABORT: a foreign ATOM server is already running."; exit 1
fi
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing: $MODEL"; exit 1; }
say "GPUs idle, no foreign server, model present"

# ---- image --------------------------------------------------------------------
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "pulling MAD image (~15.5 GB compressed); / has ${avail}G free"
  if [[ "${avail:-0}" -lt 70 ]]; then
    say "ABORT: less than 70G free on / — refusing to pull and risk filling the disk."
    say "       NOTE: the 283 GB reclaimable container is NOT ours (see plan.md) — do not prune."
    exit 1
  fi
  docker pull "$IMG" >"$OUT/pull.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "ABORT: pull failed rc=$rc"; tail -15 "$OUT/pull.log" | tee -a "$STATE"; exit 1
  fi
  say "pull OK"
else
  say "image already present"
fi
df -h / | tail -1 | tee -a "$STATE"

# ---- server -------------------------------------------------------------------
docker rm -f "$NAME" >/dev/null 2>&1
CMD="$OUT/server_cmd.sh"
cat >"$CMD" <<EOF
#!/usr/bin/env bash
set -uo pipefail

# The MAD-pinned image ships WITHOUT flash-linear-attention ('fla'), but ATOM's Kimi-K3
# model file imports it unconditionally for the KDA prefill path
# (kimi_k3.py:749 'from fla.ops.kda import chunk_kda', no flag guard, no fallback).
# 69 of Kimi-K3's 93 layers are KDA linear-attention, so without it the server loads and
# answers /v1/models, then dies on the first real request with ModuleNotFoundError.
# This is a packaging gap in the dated MAD tag -- rocm/atom-dev:latest does ship it.
# See ../notes-kimi-k3.md "Rerun attempt 1" for the full diagnosis.
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention (missing from this image)"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3
  python -c 'from fla.ops.kda import chunk_kda; print("[fix] fla.ops.kda OK")' || {
    echo "[fix] FATAL: fla still unimportable after install"; exit 1; }
else
  echo "[fix] fla already present"
fi

exec python -m atom.entrypoints.openai_server \\
  --model /model --kv_cache_dtype fp8 -tp 8 \\
  --trust-remote-code --max-model-len 16384 \\
  --max-num-seqs 64 --max-num-batched-tokens 10240 \\
  --gpu-memory-utilization 0.93 --block-size 128 \\
  --server-port $PORT \\
  --no-enable_prefix_caching 2>&1 | tee /out/atom_server.log
EOF
chmod +x "$CMD"

say "starting server (MAD env + flags) on port $PORT"
docker run -d --name "$NAME" $(dgpu_args) \
  -v "$MODEL":/model:ro -v "$OUT":/out \
  -e ATOM_LOADER_USE_THREADPOOL=1 \
  -e ATOM_LOADER_THREADPOOL_WORKERS=16 \
  -e ATOM_SYNC_AFTER_LOAD=1 \
  -e ATOM_DIST_TIMEOUT_SECONDS=3600 \
  -e ATOM_USE_TRITON_GEMM=1 \
  -e AITER_USE_GROUPED_GEMM=0 \
  -e ATOM_USE_TRITON_MOE=0 \
  -e AITER_FLYDSL_FORCE=1 \
  -e AITER_FORCE_GFX1250=0 \
  -e ATOM_USE_UNIFIED_ATTN=1 \
  -e ATOM_FORCE_ATTN_TRITON=1 \
  -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
  "$IMG" bash /out/server_cmd.sh >"$OUT/container_id.txt" 2>&1
if [[ $? -ne 0 ]]; then
  say "ABORT: docker run failed"; cat "$OUT/container_id.txt" | tee -a "$STATE"; exit 1
fi

say "waiting for server (1.5 TB load, up to 40 min)"
ready=0
for i in $(seq 1 2400); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    say "ABORT: container exited during load. Last 40 lines:"
    docker logs "$NAME" 2>&1 | tail -40 | tee -a "$STATE"
    exit 1
  fi
  (( i % 120 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
if [[ $ready -ne 1 ]]; then
  say "ABORT: server not ready in 2400s"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$STATE"
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
say "server READY"

# ---- sweep ---------------------------------------------------------------------
SUM=$OUT/summary.txt
{ echo "Kimi-K3 MAD-recipe sweep $TS"; echo "image   : $IMG";
  echo "model   : $MODEL"; echo "port    : $PORT";
  echo "ISL/OSL : $ISL/$OSL"; echo "conc    : $CONC"; echo;
  printf '%6s %12s %12s %12s %12s\n' conc req/s out_tok/s ttft_ms_med tpot_ms_med; } | tee "$SUM"

fail=0
for C in $CONC; do
  n=$(( C * 10 ))
  log="$OUT/c${C}.log"; json="$OUT/c${C}.json"
  start=$(date +%s)
  timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${PORT}" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --dataset-name random --ignore-eos \
      --request-rate inf --random-range-ratio 0.8 \
      --trust-remote-code --max-concurrency "$C" \
      --num-prompts "$n" --random-input-len "$ISL" \
      --random-output-len "$OSL" --save-result \
      --result-dir /out --result-filename "c${C}.json" >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  if [[ -f "$json" ]]; then
    read -r rps ots ttft tpot done_n < <($PY - "$json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("request_throughput",0), d.get("output_throughput",0),
      d.get("median_ttft_ms",0), d.get("median_tpot_ms",0), d.get("completed",0))
PY
)
    printf '%6s %12.2f %12.1f %12.1f %12.2f  rc=%s %ss completed=%s\n' \
      "$C" "$rps" "$ots" "$ttft" "$tpot" "$rc" "$dur" "$done_n" | tee -a "$SUM"
    if [[ "${done_n:-0}" -eq 0 ]]; then
      say "FATAL: 0 requests completed at c=$C — server rejecting. Aborting sweep."
      fail=1; break
    fi
  else
    printf '%6s %12s %12s %12s %12s  rc=%s %ss (no json)\n' "$C" - - - - "$rc" "$dur" | tee -a "$SUM"
    say "FATAL: no result json at c=$C"; fail=1; break
  fi
done

say "stopping server"
docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
sleep 5

if [[ $fail -ne 0 ]]; then
  say "SWEEP FAILED — not generating summary (a report of zeros is worse than none)."
  echo "MAD_STATUS=failed" >>"$STATE"; exit 1
fi
echo "MAD_STATUS=ok" >>"$STATE"
echo "MAD_SWEEP=$OUT" >>"$STATE"

# ---- summary --------------------------------------------------------------------
say "generating results/kimi-k3-mad.md"
$PY analyze_kimi_mad.py "$OUT" -o "$BENCH_ROOT/results" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-mad.md"
say "MAD RERUN DONE"
