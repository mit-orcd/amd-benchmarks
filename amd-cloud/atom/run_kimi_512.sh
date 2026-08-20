#!/usr/bin/env bash
# Kimi-K3 experiment: raise --max-num-seqs from 64 to 256.
#
# WHY: both prior runs capped in-flight sequences at 64 and both plateaued at ~1,180 tok/s
# while TTFT median exploded to 150 s at c=256 (see results/kimi-k3-comparison.md §2).
# That is the signature of a scheduler admission cap, not a hardware limit -- HBM sat at
# ~29%, compute ~1%, XGMI ~1%. results/kimi-k3-base.md §3.2 predicts raising the cap is the
# single biggest lever, since MoE weight traffic plateaus once all 896 experts activate,
# making extra tokens nearly free in bandwidth terms.
#
# DESIGN: one variable changed. Uses the ORIGINAL recipe/image (rocm/atom-dev:latest),
# which measured ~9% faster than the MAD recipe, with everything identical to the tier-3
# baseline EXCEPT --max-num-seqs 64 -> 256. So any difference is attributable to the cap.
#
# Memory headroom check (from kimi-k3-base.md §2): KV is 13,824 B/token/GPU; at 256 seqs x ~2048
# ctx that is ~7.2 GB against a 57.7 GB pool -- comfortable.
#
# WRITES ONLY NEW FILES: logs/atom/kimi_maxseqs_*/ and results/kimi-k3-maxseqs.{md,csv}.
# Does not touch kimi-k3-base.md, kimi-k3-mad.md, kimi-k3-comparison.md, atom.{md,csv},
# or any existing log directory.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
PORT="${PORT:-8012}"
MAXSEQS="${MAXSEQS:-512}"
CONC="${MAXSEQS_CONC:-64 128 256 512}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NAME="atom-kimi-512"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_512_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "Kimi-K3 max-num-seqs=$MAXSEQS experiment start (out: $OUT)"
say "image=$IMG conc='$CONC' ISL/OSL=$ISL/$OSL"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then say "ABORT: $busy GPU(s) busy."; exit 1; fi
if pgrep -af 'atom.entrypoints' >/dev/null 2>&1; then
  say "ABORT: a foreign ATOM server is already running."; exit 1
fi
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing: $MODEL"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: image $IMG not present"; exit 1; }
say "GPUs idle, no foreign server, model + image present"

docker rm -f "$NAME" >/dev/null 2>&1

# Original tier-3 recipe flags, verbatim, except --max-num-seqs.
KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

CMD="$OUT/server_cmd.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  echo '# rocm/atom-dev:latest ships flash-linear-attention; guard anyway so this script'
  echo '# is safe if the tag moves (the MAD-pinned image lacked it -- see notes-kimi-k3.md).'
  echo "if ! python -c 'import fla' 2>/dev/null; then"
  echo '  echo "[fix] installing flash-linear-attention"'
  echo '  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3'
  echo "  python -c 'from fla.ops.kda import chunk_kda; print(\"[fix] fla.ops.kda OK\")' || exit 1"
  echo 'else'
  echo '  echo "[fix] fla already present"'
  echo 'fi'
  echo 'exec python -m atom.entrypoints.openai_server \'
  echo '  --model /model \'
  echo '  --tensor-parallel-size 8 \'
  echo "  --server-port $PORT \\"
  echo '  --kv_cache_dtype fp8 \'
  echo "  --max-num-seqs $MAXSEQS \\"
  echo '  --gpu-memory-utilization 0.93 \'
  echo '  --trust-remote-code \'
  echo '  --max-model-len 16384 \'
  echo '  --max-num-batched-tokens 16384 \'
  echo '  --block-size 128 \'
  echo '  --no-enable_prefix_caching \'
  printf "  --online_quant_config '%s' 2>&1 | tee /out/atom_server.log\n" "$KIMI_QUANT"
} >"$CMD"
chmod +x "$CMD"

say "starting server (max-num-seqs=$MAXSEQS) on port $PORT"
docker run -d --name "$NAME" $(dgpu_args) \
  -v "$MODEL":/model:ro -v "$OUT":/out \
  -e AITER_LOG_LEVEL=WARNING \
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
    docker logs "$NAME" 2>&1 | tail -40 | tee -a "$STATE"; exit 1
  fi
  (( i % 120 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
if [[ $ready -ne 1 ]]; then
  say "ABORT: server not ready in 2400s"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$STATE"
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
say "server READY"

SUM=$OUT/summary.txt
{ echo "Kimi-K3 max-num-seqs=$MAXSEQS sweep $TS"; echo "image   : $IMG";
  echo "model   : $MODEL"; echo "port    : $PORT";
  echo "ISL/OSL : $ISL/$OSL"; echo "conc    : $CONC"; echo;
  printf '%6s %12s %12s %12s %12s\n' conc req/s out_tok/s ttft_ms_med tpot_ms_med; } | tee "$SUM"

fail=0
for C in $CONC; do
  n=$(( C * 10 ))
  log="$OUT/c${C}.log"; json="$OUT/c${C}.json"
  start=$(date +%s)
  timeout 5400 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
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
      say "FATAL: 0 requests completed at c=$C — aborting sweep."; fail=1; break
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
  say "SWEEP FAILED — not generating summary."
  echo "MAXSEQS512_STATUS=failed" >>"$STATE"; exit 1
fi
echo "MAXSEQS512_STATUS=ok" >>"$STATE"
echo "MAXSEQS512_SWEEP=$OUT" >>"$STATE"

say "generating results/kimi-k3-maxseqs512.md"
$PY analyze_kimi_512.py "$OUT" -o "$BENCH_ROOT/results" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-maxseqs512.md"
say "updating results/kimi-k3-improve.md with Run D"
$PY update_improve_with_512.py "$OUT" "$BENCH_ROOT/results/kimi-k3-improve.md" >>"$OUT/analyze.log" 2>&1
say "improve-file update rc=$?"
say "MAX-NUM-SEQS 512 EXPERIMENT DONE"
