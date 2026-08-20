#!/usr/bin/env bash
# EP retest on a FRESHLY PULLED rocm/atom-dev:latest.
#
# EP failed on :latest as of 2026-08-14 with
#   NotImplementedError: a16w4 (bf16 A x MXFP4 W) SiTUv2 is not supported: expert-parallel masking
# That tag moves. This re-pulls it and retests, closing the last open question about the
# biggest architectural lever for this model.
#
# MUST RUN LAST among Kimi stages: pulling reassigns the :latest tag, which every other
# Kimi job (profiling, 1024, 2048) depends on. Running this earlier would silently change
# the image under those runs.
#
# DISK: a fresh pull needs headroom and / is tight. Funding it by removing the MAD-pinned
# image, which by this point has already served its purpose (the MAD comparison run and the
# K2 EP retest are both complete) and is re-pullable from its dated tag. The script frees
# ONLY that image. It never touches the 283 GB foreign stopped container -- that is not ours
# (see plan.md) -- and never runs a blanket `docker system prune`.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="rocm/atom-dev:latest"
MAD_IMG="rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
PORT="${PORT:-8019}"
NEED_GB="${NEED_GB:-100}"
NAME="atom-kimi-epfresh"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_ep_fresh_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "EP retest on freshly pulled :latest — start"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model missing"; exit 1; }

# record the digest we are replacing, so the change is auditable
OLD_ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null || echo none)
say "current :latest id = ${OLD_ID:0:20}"

avail() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
say "free on / : $(avail) GB (need ~${NEED_GB} GB)"

if [[ "$(avail)" -lt "$NEED_GB" ]]; then
  say "insufficient space — removing the MAD-pinned image to make room"
  say "  (its results are already captured in kimi-k3-mad.md; the tag is re-pullable)"
  docker rmi "$MAD_IMG" >"$OUT/rmi.log" 2>&1 && say "  removed MAD image" || say "  MAD image not removable (may be absent)"
  say "free on / : $(avail) GB"
fi

if [[ "$(avail)" -lt "$NEED_GB" ]]; then
  say "ABORT: still under ${NEED_GB} GB after freeing what is safely removable."
  say "       NOT touching the 283 GB foreign container or rocm/megatron-lm (needed by N1)."
  say "       This test stays unanswered rather than risking a full disk."
  echo "EP_FRESH_STATUS=insufficient_disk" >>"$STATE"; exit 1
fi

say "pulling $IMG"
docker pull "$IMG" >"$OUT/pull.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  say "ABORT: pull failed rc=$rc"; tail -12 "$OUT/pull.log" | tee -a "$STATE"
  echo "EP_FRESH_STATUS=pull_failed" >>"$STATE"; exit 1
fi
NEW_ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null || echo none)
say "new :latest id = ${NEW_ID:0:20}"
if [[ "$OLD_ID" == "$NEW_ID" ]]; then
  say "NOTE: image digest UNCHANGED since 2026-08-14 — the tag has not moved."
  say "      An EP retest on identical bits will reproduce the same NotImplementedError."
  say "      Running it anyway is ~10 min and makes the negative result explicit."
  echo "EP_FRESH_IMAGE=unchanged" >>"$STATE"
else
  say "image HAS changed since the original EP test — retest is meaningful"
  echo "EP_FRESH_IMAGE=changed" >>"$STATE"
fi

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
  echo '  --enable-expert-parallel \'
  printf "  --online_quant_config '%s' 2>&1 | tee /out/atom_server.log\n" "$KIMI_QUANT"
} >"$CMD"
chmod +x "$CMD"

say "starting server with --enable-expert-parallel"
docker run -d --name "$NAME" $(dgpu_args) -v "$MODEL":/model:ro -v "$OUT":/out \
  -e AITER_LOG_LEVEL=WARNING -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
  "$IMG" bash /out/server_cmd.sh >"$OUT/cid.txt" 2>&1 || { say "ABORT: docker run failed"; exit 1; }

say "waiting (EP failure, if any, appears at load)"
ready=0
for i in $(seq 1 2400); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    say "=== EP STILL NOT SUPPORTED on :latest ==="
    if docker logs "$NAME" 2>&1 | grep -qi 'expert-parallel masking'; then
      say "same failure: SiTUv2 expert-parallel masking NotImplementedError"
      echo "EP_FRESH_STATUS=same_notimplemented" >>"$STATE"
    else
      echo "EP_FRESH_STATUS=other_failure" >>"$STATE"
    fi
    docker logs "$NAME" 2>&1 | grep -iE 'error|notimplemented|traceback' | tail -8 | tee -a "$STATE"
    docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  (( i % 180 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
[[ $ready -eq 1 ]] || { say "ABORT: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1
  echo "EP_FRESH_STATUS=timeout" >>"$STATE"; exit 1; }

say "=== EP WORKS on the new :latest — the HBM->XGMI trade is now testable ==="
echo "EP_FRESH_STATUS=works" >>"$STATE"

SUM=$OUT/summary.txt
{ echo "Kimi-K3 EP-enabled sweep (fresh :latest) $TS"; echo;
  printf '%6s %12s %12s %12s\n' conc out_tok/s ttft_ms_med tpot_ms_med; } | tee "$SUM"
for C in ${EP_CONC:-64 128 256}; do
  timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
    --model /model --backend vllm --base-url "http://localhost:${PORT}" \
    --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
    --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
    --max-concurrency "$C" --num-prompts $(( C * 10 )) \
    --random-input-len 1024 --random-output-len 1024 --save-result \
    --result-dir /out --result-filename "c${C}.json" >"$OUT/c${C}.log" 2>&1
  if [[ -f "$OUT/c${C}.json" ]]; then
    printf '%6s %12s %12s %12s\n' "$C" \
      "$($PY -c "import json;print('%.1f'%json.load(open('$OUT/c${C}.json'))['output_throughput'])")" \
      "$($PY -c "import json;print('%.1f'%json.load(open('$OUT/c${C}.json'))['median_ttft_ms'])")" \
      "$($PY -c "import json;print('%.2f'%json.load(open('$OUT/c${C}.json'))['median_tpot_ms'])")" \
      | tee -a "$SUM"
  fi
done
docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
say "EP FRESH RETEST DONE"
