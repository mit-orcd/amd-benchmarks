#!/usr/bin/env bash
# EP retest on the MAD-pinned image.
#
# EP failed on rocm/atom-dev:latest (2026-08-14) with
#   NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel masking
# and was never retried elsewhere. Testing it here rather than on a freshly pulled :latest
# for two reasons: the MAD image is ALREADY LOCAL (a fresh :latest pull is ~106 GB and / has
# only ~61 GB free), and MAD sets ATOM_USE_TRITON_MOE=0 / AITER_USE_GROUPED_GEMM=0, which
# select a DIFFERENT MoE kernel path -- and the failure came from the SiTUv2 kernel
# specifically. So this is the cheap test with a real mechanism behind it.
#
# Cheap by design: the failure surfaces at model load, so this either dies in ~10 min or
# proves EP works and runs a short sweep. Writes only new files.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="${MAD_IMG:-rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
PORT="${PORT:-8017}"
CONC="${EP_CONC:-64 128}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NAME="atom-kimi-epmad"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_ep_mad_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "EP retest on MAD image start (out: $OUT)"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server."; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: MAD image absent"; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1

CMD="$OUT/server_cmd.sh"
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
  echo '  --kv_cache_dtype fp8 --max-num-seqs 256 \'
  echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
  echo '  --max-model-len 16384 --max-num-batched-tokens 10240 \'
  echo '  --block-size 128 --no-enable_prefix_caching \'
  echo '  --enable-expert-parallel 2>&1 | tee /out/atom_server.log'
} >"$CMD"
chmod +x "$CMD"

say "starting server with --enable-expert-parallel (MAD env + MAD image)"
docker run -d --name "$NAME" $(dgpu_args) \
  -v "$MODEL":/model:ro -v "$OUT":/out \
  -e ATOM_LOADER_USE_THREADPOOL=1 -e ATOM_LOADER_THREADPOOL_WORKERS=16 \
  -e ATOM_SYNC_AFTER_LOAD=1 -e ATOM_DIST_TIMEOUT_SECONDS=3600 \
  -e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0 -e ATOM_USE_TRITON_MOE=0 \
  -e AITER_FLYDSL_FORCE=1 -e AITER_FORCE_GFX1250=0 \
  -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1 \
  -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
  "$IMG" bash /out/server_cmd.sh >"$OUT/cid.txt" 2>&1 || { say "ABORT: docker run failed"; exit 1; }

say "waiting for server (EP failure, if any, appears at load)"
ready=0
for i in $(seq 1 2400); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    say "=== EP NOT SUPPORTED ON MAD IMAGE EITHER ==="
    if docker logs "$NAME" 2>&1 | grep -qi 'expert-parallel masking'; then
      say "same failure: SiTUv2 expert-parallel masking NotImplementedError"
      echo "EP_MAD_STATUS=same_notimplemented" >>"$STATE"
    else
      say "container died for a different reason:"
      echo "EP_MAD_STATUS=other_failure" >>"$STATE"
    fi
    docker logs "$NAME" 2>&1 | grep -iE 'error|notimplemented|traceback' | tail -8 | tee -a "$STATE"
    docker rm -f "$NAME" >/dev/null 2>&1
    exit 1
  fi
  (( i % 180 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
if [[ $ready -ne 1 ]]; then
  say "ABORT: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1
  echo "EP_MAD_STATUS=timeout" >>"$STATE"; exit 1
fi

say "=== EP WORKS ON THE MAD IMAGE — this contradicts the :latest result ==="
echo "EP_MAD_STATUS=works" >>"$STATE"

SUM=$OUT/summary.txt
{ echo "Kimi-K3 EP-enabled sweep (MAD image) $TS"; echo;
  printf '%6s %12s %12s %12s\n' conc out_tok/s ttft_ms_med tpot_ms_med; } | tee "$SUM"
for C in $CONC; do
  timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
    --model /model --backend vllm --base-url "http://localhost:${PORT}" \
    --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
    --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
    --max-concurrency "$C" --num-prompts $(( C * 10 )) \
    --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
    --result-dir /out --result-filename "c${C}.json" >"$OUT/c${C}.log" 2>&1
  if [[ -f "$OUT/c${C}.json" ]]; then
    printf '%6s %12s %12s %12s\n' "$C" \
      "$($PY -c "import json;print('%.1f'%json.load(open('$OUT/c${C}.json'))['output_throughput'])")" \
      "$($PY -c "import json;print('%.1f'%json.load(open('$OUT/c${C}.json'))['median_ttft_ms'])")" \
      "$($PY -c "import json;print('%.2f'%json.load(open('$OUT/c${C}.json'))['median_tpot_ms'])")" \
      | tee -a "$SUM"
  else
    say "c=$C produced no json"
  fi
done

docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
say "EP MAD RETEST DONE"
