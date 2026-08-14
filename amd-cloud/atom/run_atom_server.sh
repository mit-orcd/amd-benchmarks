#!/usr/bin/env bash
# Start the ATOM OpenAI-compatible server in a container.
#
# Usage: ./run_atom_server.sh <model_dir> [TP] [PORT] [extra atom args...]
#   ./run_atom_server.sh $SCRATCH/models/Qwen3-8B-FP8 1 8000
#
# Deliberately NOT a wrapper around ATOM's own scripts/start_atom_server.sh. That script
# opens with `pkill -f 'atom.entrypoints'` and `pkill -9 -f 'multiprocessing.spawn'`, which
# on this shared box could kill another user's inference server, and it hardcodes
# KINETO_CONFIG to a path in someone else's home directory. This one refuses instead of
# killing, and never touches processes it does not own.
#
# Runs the server in the FOREGROUND of its container; the container is detached, so the
# script returns once the server is up. Stop it with ./stop_atom_server.sh.
#
# Entrypoint/flag notes -- these were verified against the source, not guessed:
#   * `atom.entrypoints.openai_server` (NOT `atom.entrypoints.openai.api_server`). The
#     former is a thin wrapper that calls set_ulimit() before api_server.main(), raising
#     the open-file soft limit. Without it, high-concurrency sweeps exhaust file
#     descriptors. ATOM's own scripts and every recipe use this module.
#   * `--kv_cache_dtype` is spelled with UNDERSCORES; `--kv-cache-dtype` does not exist.
#   * The port flag is `--server-port`; `--port` does not exist.
#   * `--tensor-parallel-size` and `-tp` are aliases; either works.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh

MODEL="${1:?usage: $0 <model_dir> [TP] [PORT] [extra args...]}"
TP="${2:-1}"
PORT="${3:-8000}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$*"

IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
NAME="${ATOM_CONTAINER:-atom-bench}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/server_$TS; mkdir -p "$OUT"

[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL" >&2; exit 1; }

# ---- safety: never stomp on another user's work --------------------------------
if pgrep -af 'atom.entrypoints' >/dev/null 2>&1; then
  echo "REFUSING TO START: an ATOM server process is already running on this host." >&2
  pgrep -af 'atom.entrypoints' >&2
  echo "This box is shared (rocm/atom-dev predates us). Stop it deliberately, or set a" >&2
  echo "different port and re-run, but do NOT pkill it blindly." >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "REFUSING TO START: container '$NAME' is already running. ./stop_atom_server.sh first." >&2
  exit 1
fi
busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  echo "REFUSING TO START: $busy GPU(s) busy — another benchmark is running." >&2
  echo "All parts of this plan are strictly sequential; wait for it to finish." >&2
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "REFUSING TO START: port $PORT already in use." >&2
  exit 1
fi

devs=$(seq -s, 0 $((TP-1)))
echo "Starting ATOM server"
echo "  image  : $IMG"
echo "  model  : $MODEL"
echo "  TP     : $TP  (devices $devs)"
echo "  port   : $PORT"
echo "  logs   : $OUT"

# Build the launch command as a FILE rather than inlining it into `bash -c`.
# Kimi-K3 needs --online_quant_config with a JSON argument full of double quotes and
# spaces; nesting that through docker + bash -c quoting is how you get a silently
# mangled flag. Writing a script and mounting it sidesteps quoting entirely.
CMD="$OUT/server_cmd.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  echo 'exec python -m atom.entrypoints.openai_server \'
  echo '  --model /model \'
  # NO --served-model-name. The benchmark client (run_atom_bench.sh) passes
  # `--model=/model` because it also loads the tokenizer from that path, and it sends that
  # same string as the OpenAI `model` field. If the server registers a different id
  # (e.g. "atom-bench"), every request comes back HTTP 400 while /v1/models and the health
  # endpoints still answer 200 -- so the readiness check passes and the whole sweep silently
  # records 0.00 for every metric. Serving under the path keeps client and server aligned.
  echo "  --tensor-parallel-size $TP \\"
  echo "  --server-port $PORT \\"
  echo "  --kv_cache_dtype $KV_CACHE_DTYPE \\"
  echo "  --max-num-seqs $MAX_NUM_SEQS \\"
  echo "  --gpu-memory-utilization $GPU_MEM_UTIL \\"
  echo '  --trust-remote-code \'
  printf '  %s 2>&1 | tee /out/atom_server.log\n' "$EXTRA_ARGS"
} >"$CMD"
chmod +x "$CMD"
echo "  cmd    : $CMD"

docker run -d --name "$NAME" $(dgpu_args) \
  -v "$MODEL":/model:ro \
  -v "$OUT":/out \
  -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
  -e AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}" \
  -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
  "$IMG" bash /out/server_cmd.sh >"$OUT/container_id.txt" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "ERROR: docker run failed (rc=$rc)" >&2
  cat "$OUT/container_id.txt" >&2
  exit 1
fi
echo "  container: $(cut -c1-12 <"$OUT/container_id.txt")"

# ---- wait for ready: HTTP + VRAM actually allocated ----------------------------
echo -n "Waiting for server"
for i in $(seq 1 "${READY_TIMEOUT:-600}"); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    vram=$(rocm-smi --showmemuse 2>/dev/null | grep -c 'VRAM%.*[1-9]' || echo 0)
    if [[ "${vram:-0}" -gt 0 ]]; then
      echo; echo "Server READY on port $PORT (VRAM loaded on $vram GPU(s))"
      echo "$PORT" > "$LOG_ROOT/atom/CURRENT_PORT.txt"
      echo "$OUT"  > "$LOG_ROOT/atom/CURRENT_SERVER_DIR.txt"
      exit 0
    fi
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo; echo "ERROR: container exited before becoming ready. Last 40 lines:" >&2
    docker logs "$NAME" 2>&1 | tail -40 >&2
    exit 1
  fi
  (( i % 15 == 0 )) && echo -n " ${i}s"
  sleep 1
done
echo; echo "ERROR: server not ready within ${READY_TIMEOUT:-600}s. Last 40 lines:" >&2
docker logs "$NAME" 2>&1 | tail -40 >&2
exit 1
