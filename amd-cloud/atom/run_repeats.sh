#!/usr/bin/env bash
# Next-step #4: repeats at c=64, to settle whether the "MAD recipe is ~9% slower" finding
# is real or run-to-run noise. That claim currently rests on ONE run per config, which
# supports no conclusion about a difference that size.
#
# DESIGN: 3 repeats at c=64 on each config. The model is loaded ONCE per config and the
# benchmark run 3x against the same live server -- 2 loads instead of 6, saving ~35 min.
# That does mean repeats share a server process, so this measures benchmark-to-benchmark
# variance, not full cold-start variance. Stated in the output rather than glossed over.
#
# Config A: rocm/atom-dev:latest + original recipe (--online_quant_config, batched 16384)
# Config B: MAD-pinned image + 11 MAD env vars, batched 10240, no online_quant_config
# Both at --max-num-seqs 64, c=64, ISL/OSL 1024/1024 -- matching the original comparison.
#
# WRITES ONLY NEW FILES: logs/atom/kimi_repeats_*/ and results/kimi-k3-repeats.{md,csv}.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

REPS="${REPS:-3}"
CONC="${REPEAT_CONC:-64}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
IMG_A="${IMG_A:-rocm/atom-dev:latest}"
IMG_B="${IMG_B:-rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3}"
PORT="${PORT:-8014}"
NAME="atom-kimi-repeats"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_repeats_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "Kimi-K3 repeatability run start (reps=$REPS, c=$CONC) out: $OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server running."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing"; exit 1; }
say "GPUs idle, model present"

KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

run_config() {
  local tag=$1 img=$2
  say "===== config $tag : $img ====="
  docker image inspect "$img" >/dev/null 2>&1 || { say "SKIP $tag: image absent"; return 1; }
  docker rm -f "$NAME" >/dev/null 2>&1

  local CMD="$OUT/${tag}_server_cmd.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo "if ! python -c 'import fla' 2>/dev/null; then"
    echo '  echo "[fix] installing flash-linear-attention"'
    echo '  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -2'
    echo "  python -c 'from fla.ops.kda import chunk_kda' || exit 1"
    echo 'fi'
    echo 'exec python -m atom.entrypoints.openai_server \'
    echo '  --model /model --tensor-parallel-size 8 \'
    echo "  --server-port $PORT \\"
    echo '  --kv_cache_dtype fp8 --max-num-seqs 64 \'
    echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
    echo '  --max-model-len 16384 --block-size 128 --no-enable_prefix_caching \'
    if [[ "$tag" == "A_original" ]]; then
      echo '  --max-num-batched-tokens 16384 \'
      printf "  --online_quant_config '%s' 2>&1 | tee /out/%s_server.log\n" "$KIMI_QUANT" "$tag"
    else
      printf '  --max-num-batched-tokens 10240 2>&1 | tee /out/%s_server.log\n' "$tag"
    fi
  } >"$CMD"
  chmod +x "$CMD"

  local ENVS=(-e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN)
  if [[ "$tag" == "B_mad" ]]; then
    ENVS+=(-e ATOM_LOADER_USE_THREADPOOL=1 -e ATOM_LOADER_THREADPOOL_WORKERS=16
           -e ATOM_SYNC_AFTER_LOAD=1 -e ATOM_DIST_TIMEOUT_SECONDS=3600
           -e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0 -e ATOM_USE_TRITON_MOE=0
           -e AITER_FLYDSL_FORCE=1 -e AITER_FORCE_GFX1250=0
           -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1)
  else
    ENVS+=(-e AITER_LOG_LEVEL=WARNING)
  fi

  docker run -d --name "$NAME" $(dgpu_args) -v "$MODEL":/model:ro -v "$OUT":/out \
    "${ENVS[@]}" "$img" bash "/out/${tag}_server_cmd.sh" >"$OUT/${tag}_cid.txt" 2>&1 || {
      say "$tag: docker run failed"; return 1; }

  say "$tag: waiting for server"
  local ready=0
  for i in $(seq 1 2400); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
      say "$tag: container died during load"; docker logs "$NAME" 2>&1 | tail -25 | tee -a "$STATE"
      return 1; }
    (( i % 180 == 0 )) && say "  $tag still loading (${i}s)"
    sleep 1
  done
  [[ $ready -eq 1 ]] || { say "$tag: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1; return 1; }
  say "$tag: server READY"

  for r in $(seq 1 "$REPS"); do
    local json="$OUT/${tag}_rep${r}.json"
    timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${PORT}" \
      --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
      --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
      --max-concurrency "$CONC" --num-prompts $(( CONC * 10 )) \
      --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
      --result-dir /out --result-filename "${tag}_rep${r}.json" \
      >"$OUT/${tag}_rep${r}.log" 2>&1
    if [[ -f "$json" ]]; then
      say "  $tag rep$r: $($PY -c "import json;d=json.load(open('$json'));print('%.1f tok/s completed=%d'%(d['output_throughput'],d['completed']))")"
    else
      say "  $tag rep$r: FAILED (no json)"
    fi
  done

  docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  sleep 5
  say "$tag: done"
}

run_config A_original "$IMG_A" || say "config A incomplete"
run_config B_mad      "$IMG_B" || say "config B incomplete"

n=$(ls "$OUT"/*_rep*.json 2>/dev/null | wc -l)
say "collected $n result files"
if [[ "$n" -eq 0 ]]; then
  say "NO RESULTS — not generating summary."; echo "REPEATS_STATUS=failed" >>"$STATE"; exit 1
fi
echo "REPEATS_STATUS=ok" >>"$STATE"; echo "REPEATS_SWEEP=$OUT" >>"$STATE"

say "generating results/kimi-k3-repeats.md"
$PY analyze_repeats.py "$OUT" -o "$BENCH_ROOT/results" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-repeats.md"
say "REPEATS RUN DONE"
