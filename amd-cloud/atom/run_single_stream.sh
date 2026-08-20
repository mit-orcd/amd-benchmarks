#!/usr/bin/env bash
# Single-stream (per-request) speed experiment -- the first one here that targets TPOT
# instead of aggregate throughput.
#
# WHY: every Kimi-K3 experiment so far measured aggregate tok/s, and each either left
# per-request speed unchanged (~20 tok/s at c=64 across runs A, C and D) or made it worse by
# admitting a bigger batch. Run D's headline 3,385.9 tok/s at c=512 is simultaneously the
# WORST single-user experience measured: 6.5 tok/s. The single-stream number rests on one
# measurement -- 46.6 tok/s at c=1 (TPOT 21.48 ms) in Run A -- never reproduced, never tested
# against a different kernel configuration.
#
# THE HEADROOM: at c=1 a decode step reads ~3.4 GB of weights per GPU. At the ~2.2 TB/s
# effective rate implied by c=64 (116 GB / 51.7 ms) that is ~1.5 ms of weight reading against
# a measured 21.48 ms step -- so ~93% of a single-request step is NOT weight traffic. It is
# serialization (186 all-reduces, 69 sequential KDA layers, kernel launch overhead), and
# unlike a bandwidth wall that is compressible.
#
# WHAT THIS CAN AND CANNOT TEST. Of the levers that could move per-request speed, only kernel
# path selection is reachable from configuration:
#   - collective count   : fixed by TP=8 x 93 layers, no flag
#   - TP < 8 + replicas  : 1.5 TB of weights vs 2.3 TB HBM forces TP=8 on one node
#   - speculative/MTP    : num_nextn_predict_layers = 0, no MTP heads shipped
#   - HIP graphs         : already on (server log reports cudagraph=True)
# So this sweeps kernel paths at LOW concurrency, where kernel efficiency and launch overhead
# dominate -- every previous kernel comparison ran at c=64+, which hides exactly those costs.
#
# DESIGN: four arms, each flipping ONE kernel-path decision from the MAD baseline, so any
# difference is attributable. K1 is the unmodified MAD set and doubles as the matched control
# for Run A's 46.6 tok/s (Run A used rocm/atom-dev:latest; these run on the MAD image, so
# comparing an arm directly to 46.6 would confound image with kernel path).
#
# EXPECTATIONS, STATED IN ADVANCE: modest effects, single-digit to low-tens of percent. The
# MAD set was already ~9% slower in aggregate at c=64. A NEGATIVE result is still worth
# having: it localizes the 93% to launch overhead and the KDA dependency chain -- things no
# env var can reach -- and closes configuration-level tuning for this model.
#
# NOT AUTO-STARTED. Run explicitly, or chain it from a queue driver.
# WRITES ONLY NEW FILES: logs/atom/kimi_single_stream_*/ and results/kimi-k3-single-stream.md.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="${MAD_IMG:-rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
CONC="${SS_CONC:-1 2 4 8}"
MAXSEQS="${MAXSEQS:-64}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
PROMPT_MULT="${PROMPT_MULT:-5}"   # num-prompts = max(8, C * PROMPT_MULT)
PORT="${PORT:-8022}"
NAME="atom-kimi-ss"
ARMS="${SS_ARMS:-K1_mad_default K2_triton_moe K3_aiter_attn K4_grouped_gemm}"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_single_stream_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "single-stream experiment start (conc='$CONC', arms='$ARMS') out: $OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || {
  say "ABORT: MAD image absent. Pull it first (~64 GB) -- this script does not free disk"
  say "       or pull on its own, so it can never surprise a running job by moving a tag."
  echo "SS_STATUS=image_absent" >>"$STATE"; exit 1; }

# MAD baseline kernel env. Each arm overrides exactly one decision.
arm_env() {
  local arm=$1
  local -n _out=$2
  _out=(-e ATOM_LOADER_USE_THREADPOOL=1 -e ATOM_LOADER_THREADPOOL_WORKERS=16
        -e ATOM_SYNC_AFTER_LOAD=1 -e ATOM_DIST_TIMEOUT_SECONDS=3600
        -e AITER_FLYDSL_FORCE=1 -e AITER_FORCE_GFX1250=0
        -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN)
  case "$arm" in
    K1_mad_default)   _out+=(-e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0
                             -e ATOM_USE_TRITON_MOE=0
                             -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1) ;;
    K2_triton_moe)    _out+=(-e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0
                             -e ATOM_USE_TRITON_MOE=1
                             -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1) ;;
    K3_aiter_attn)    _out+=(-e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0
                             -e ATOM_USE_TRITON_MOE=0
                             -e ATOM_USE_UNIFIED_ATTN=0 -e ATOM_FORCE_ATTN_TRITON=0) ;;
    K4_grouped_gemm)  _out+=(-e ATOM_USE_TRITON_GEMM=0 -e AITER_USE_GROUPED_GEMM=1
                             -e ATOM_USE_TRITON_MOE=0
                             -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1) ;;
    *) say "unknown arm $arm"; return 1 ;;
  esac
}

run_arm() {
  local arm=$1
  local ARM_OUT="$OUT/$arm"; mkdir -p "$ARM_OUT"
  say "===== arm $arm ====="
  docker rm -f "$NAME" >/dev/null 2>&1

  local CMD="$ARM_OUT/server_cmd.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo "if ! python -c 'import fla' 2>/dev/null; then"
    echo '  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -2'
    echo "  python -c 'from fla.ops.kda import chunk_kda' || exit 1"
    echo 'fi'
    echo 'exec python -m atom.entrypoints.openai_server \'
    echo '  --model /model --tensor-parallel-size 8 \'
    echo "  --server-port $PORT \\"
    echo "  --kv_cache_dtype fp8 --max-num-seqs $MAXSEQS \\"
    echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
    echo '  --max-model-len 16384 --max-num-batched-tokens 10240 \'
    echo '  --block-size 128 --no-enable_prefix_caching \'
    echo '  2>&1 | tee /out/atom_server.log'
  } >"$CMD"
  chmod +x "$CMD"

  local ENVS
  arm_env "$arm" ENVS || return 1
  printf '%s\n' "${ENVS[@]}" >"$ARM_OUT/env.txt"

  docker run -d --name "$NAME" $(dgpu_args) -v "$MODEL":/model:ro -v "$ARM_OUT":/out \
    "${ENVS[@]}" "$IMG" bash /out/server_cmd.sh >"$ARM_OUT/cid.txt" 2>&1 || {
      say "$arm: docker run failed"; return 1; }

  say "$arm: waiting for server"
  local ready=0
  for i in $(seq 1 2400); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
      say "$arm: container died during load (this arm's kernel set may be unsupported)"
      docker logs "$NAME" 2>&1 | grep -iE 'error|notimplemented|traceback' | tail -10 | tee -a "$STATE"
      docker rm -f "$NAME" >/dev/null 2>&1; return 1; }
    (( i % 300 == 0 )) && say "  $arm still loading (${i}s)"
    sleep 1
  done
  [[ $ready -eq 1 ]] || { say "$arm: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1; return 1; }
  say "$arm: server READY"

  for C in $CONC; do
    local NP=$(( C * PROMPT_MULT )); (( NP < 8 )) && NP=8
    timeout 5400 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${PORT}" \
      --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
      --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
      --max-concurrency "$C" --num-prompts "$NP" \
      --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
      --result-dir /out --result-filename "c${C}.json" >"$ARM_OUT/c${C}.log" 2>&1
    if [[ -f "$ARM_OUT/c${C}.json" ]]; then
      say "  $arm c=$C: $($PY -c "
import json;d=json.load(open('$ARM_OUT/c${C}.json'))
t=d['median_tpot_ms'];print('%.2f ms TPOT -> %.1f tok/s per request (aggregate %.1f)'%(t,1000.0/t,d['output_throughput']))")"
    else
      say "  $arm c=$C: FAILED (no json)"
    fi
  done

  docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  sleep 10
  say "$arm: done"
}

for arm in $ARMS; do
  run_arm "$arm" || say "arm $arm incomplete — continuing"
done

n=$(ls "$OUT"/*/c*.json 2>/dev/null | wc -l)
say "collected $n result files across arms"
echo "SS_N=$n" >>"$STATE"
if [[ "$n" -eq 0 ]]; then
  say "NO RESULTS — not generating the report"; echo "SS_STATUS=failed" >>"$STATE"; exit 1
fi
echo "SS_STATUS=ok" >>"$STATE"; echo "SS_SWEEP=$OUT" >>"$STATE"

say "generating results/kimi-k3-single-stream.md"
$PY analyze_single_stream.py "$OUT" -o "$BENCH_ROOT/results" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-single-stream.md"

say "folding result into results/kimi-k3-improve.md"
$PY update_improve_with_ss.py "$OUT" "$BENCH_ROOT/results/kimi-k3-improve.md" \
   >"$OUT/update_improve.log" 2>&1
say "improve-file update rc=$? -> results/kimi-k3-improve.md"
say "SINGLE-STREAM EXPERIMENT DONE"
