#!/usr/bin/env bash
# Next-step #2: capture a real profiler trace of a decode step.
#
# WHY: every statement in kimi-k3-improve.md about where the non-weight ~80% of step time
# goes is inference from residuals -- we subtract estimated weight/KV/compute/collective
# costs and attribute the remainder to "prefill + scheduling". That is a guess. A trace
# replaces it with measurement, and it is the prerequisite for judging any kernel-level
# idea (including whether EP would ever be worth it).
#
# Runs at c=256 on the Run C config (cap 256) -- a known-good, already-characterised point,
# so the trace is directly comparable to numbers already in the report.
#
# Traces go to /mnt/scratch (NOT the repo): torch traces are routinely hundreds of MB and
# *.trace.json is gitignored anyway. Only the derived summary lands in results/.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
PORT="${PORT:-8015}"
CONC="${PROF_CONC:-256}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NAME="atom-kimi-prof"
TRACE_DIR="$SCRATCH/traces/kimi_$(date +%Y%m%d_%H%M%S)"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_profile_$TS; mkdir -p "$OUT" "$TRACE_DIR"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "Kimi-K3 profiling run start (c=$CONC) out: $OUT traces: $TRACE_DIR"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model missing"; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1

KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

CMD="$OUT/server_cmd.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  echo "if ! python -c 'import fla' 2>/dev/null; then pip install --no-cache-dir flash-linear-attention 2>&1|tail -2; fi"
  echo 'exec python -m atom.entrypoints.openai_server \'
  echo '  --model /model --tensor-parallel-size 8 \'
  echo "  --server-port $PORT \\"
  echo '  --kv_cache_dtype fp8 --max-num-seqs 256 \'
  echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
  echo '  --max-model-len 16384 --max-num-batched-tokens 16384 \'
  echo '  --block-size 128 --no-enable_prefix_caching \'
  echo '  --torch-profiler-dir /traces \'
  printf "  --online_quant_config '%s' 2>&1 | tee /out/atom_server.log\n" "$KIMI_QUANT"
} >"$CMD"
chmod +x "$CMD"

say "starting server with --torch-profiler-dir"
docker run -d --name "$NAME" $(dgpu_args) \
  -v "$MODEL":/model:ro -v "$OUT":/out -v "$TRACE_DIR":/traces \
  -e AITER_LOG_LEVEL=WARNING \
  -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
  "$IMG" bash /out/server_cmd.sh >"$OUT/cid.txt" 2>&1 || { say "ABORT: docker run failed"; exit 1; }

say "waiting for server"
ready=0
for i in $(seq 1 2400); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
    say "ABORT: container died during load"; docker logs "$NAME" 2>&1|tail -30|tee -a "$STATE"; exit 1; }
  (( i % 180 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
[[ $ready -eq 1 ]] || { say "ABORT: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
say "server READY"

# Short profiled run: --profile makes benchmark_serving trigger start/stop around the run.
# Keep num-prompts small -- a trace of a few hundred steps is plenty and keeps the file sane.
say "running profiled benchmark (c=$CONC, short)"
timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
  --model /model --backend vllm --base-url "http://localhost:${PORT}" \
  --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
  --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
  --max-concurrency "$CONC" --num-prompts "$CONC" \
  --random-input-len "$ISL" --random-output-len "$OSL" \
  --profile --save-result --result-dir /out --result-filename profile_run.json \
  >"$OUT/profile_bench.log" 2>&1
rc=$?
say "profiled bench rc=$rc"

say "stopping server"
docker stop -t 60 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
sleep 10

ntr=$(find "$TRACE_DIR" -name '*.json*' 2>/dev/null | wc -l)
sz=$(du -sh "$TRACE_DIR" 2>/dev/null | cut -f1)
say "traces captured: $ntr file(s), $sz in $TRACE_DIR"
echo "TRACE_DIR=$TRACE_DIR" >>"$STATE"
echo "TRACE_FILES=$ntr" >>"$STATE"

if [[ "$ntr" -eq 0 ]]; then
  say "NO TRACES — profiling did not produce output. Last 25 lines of bench log:"
  tail -25 "$OUT/profile_bench.log" | tee -a "$STATE"
  echo "PROFILE_STATUS=no_traces" >>"$STATE"; exit 1
fi
echo "PROFILE_STATUS=ok" >>"$STATE"

say "summarising trace with analyze_profile.py"
$PY analyze_profile.py "$TRACE_DIR" "$OUT" -o "$BENCH_ROOT/results" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-profile.md"
say "PROFILING RUN DONE"
