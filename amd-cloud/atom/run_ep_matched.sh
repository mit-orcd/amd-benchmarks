#!/usr/bin/env bash
# Matched-max-num-seqs EP vs TP-only A/B on the MAD image.
#
# WHY: the 2026-08-20 EP result compared an EP run at --max-num-seqs 256 against a TP-only
# baseline at --max-num-seqs 64. Only the c=64 row was a valid comparison (both under cap);
# the c=128 row was confounded by the admission cap, which section 4 already showed is worth
# 2.1x on its own. On the one clean point EP cost 14% -- but c=64 is the regime where EP is
# LEAST likely to pay off, since EP's per-GPU weight-read saving grows with batch size while
# its all-to-all latency stays roughly fixed. This runs BOTH arms at an identical cap of 256
# across c=64/128/256 so the comparison is clean at a batch where EP could actually win.
#
# EP only loads on the MAD image (rocm/atom-dev:latest raises NotImplementedError in the
# SiTUv2 kernel; confirmed again on a freshly pulled :latest 2026-08-20). MAD sets
# ATOM_USE_TRITON_MOE=0 / AITER_USE_GROUPED_GEMM=0, selecting an MoE kernel with no such
# restriction. Both arms therefore use the MAD image + MAD env vars; the ONLY difference
# between them is --enable-expert-parallel.
#
# DISK: K5 removed the MAD image to fund its :latest pull, so it must be re-pulled. / has
# ~61 GB free and the image is ~64 GB. Funding it by removing rocm/atom-dev:latest, which no
# remaining queued job needs (every :latest-based Kimi run -- base, caps 256/512/1024,
# ISL=4096, profiling, repeats config A -- is complete) and which is re-pullable in ~3 min.
# Never touches the 283 GB foreign stopped container (not ours, see plan.md) or
# rocm/megatron-lm:v26.1 (needed by the megatron-ref rerun that follows).
#
# WRITES ONLY NEW FILES: logs/atom/kimi_ep_matched_*/ and results/kimi-k3-ep-matched.md.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

MAD_IMG="${MAD_IMG:-rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3}"
LATEST_IMG="rocm/atom-dev:latest"
MODEL="${MKIMI:-$SCRATCH/models/Kimi-K3}"
MAXSEQS="${MAXSEQS:-256}"
CONC="${EPM_CONC:-64 128 256}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NEED_GB="${NEED_GB:-80}"
NAME="atom-kimi-epm"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_ep_matched_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "matched-cap EP A/B start (cap=$MAXSEQS, conc='$CONC') out: $OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: foreign ATOM server."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing"; exit 1; }

avail() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

if ! docker image inspect "$MAD_IMG" >/dev/null 2>&1; then
  say "MAD image absent (K5 removed it) — need to re-pull"
  say "free on / : $(avail) GB (need ~${NEED_GB} GB)"
  if [[ "$(avail)" -lt "$NEED_GB" ]]; then
    say "removing $LATEST_IMG — no remaining queued job needs it, re-pullable in ~3 min"
    docker rmi "$LATEST_IMG" >"$OUT/rmi.log" 2>&1 && say "  removed" || say "  not removable"
    say "free on / : $(avail) GB"
  fi
  if [[ "$(avail)" -lt "$NEED_GB" ]]; then
    say "ABORT: still under ${NEED_GB} GB after freeing what is safely removable."
    say "       NOT touching the 283 GB foreign container or rocm/megatron-lm (megatron-ref needs it)."
    echo "EPM_STATUS=insufficient_disk" >>"$STATE"; exit 1
  fi
  say "pulling $MAD_IMG"
  docker pull "$MAD_IMG" >"$OUT/pull.log" 2>&1 || {
    say "ABORT: pull failed"; tail -10 "$OUT/pull.log" | tee -a "$STATE"
    echo "EPM_STATUS=pull_failed" >>"$STATE"; exit 1; }
  say "pull ok"
fi

# ---- one arm: $1 = tag (tp_only|ep), $2 = extra server flag, $3 = port ----
run_arm() {
  local tag=$1 extra=$2 port=$3
  local ARM_OUT="$OUT/$tag"; mkdir -p "$ARM_OUT"
  say "===== arm $tag (cap=$MAXSEQS, extra='${extra:-none}') ====="
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
    echo "  --server-port $port \\"
    echo "  --kv_cache_dtype fp8 --max-num-seqs $MAXSEQS \\"
    echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
    echo '  --max-model-len 16384 --max-num-batched-tokens 10240 \'
    echo '  --block-size 128 --no-enable_prefix_caching \'
    [[ -n "$extra" ]] && echo "  $extra \\"
    echo '  2>&1 | tee /out/atom_server.log'
  } >"$CMD"
  chmod +x "$CMD"

  docker run -d --name "$NAME" $(dgpu_args) -v "$MODEL":/model:ro -v "$ARM_OUT":/out \
    -e ATOM_LOADER_USE_THREADPOOL=1 -e ATOM_LOADER_THREADPOOL_WORKERS=16 \
    -e ATOM_SYNC_AFTER_LOAD=1 -e ATOM_DIST_TIMEOUT_SECONDS=3600 \
    -e ATOM_USE_TRITON_GEMM=1 -e AITER_USE_GROUPED_GEMM=0 -e ATOM_USE_TRITON_MOE=0 \
    -e AITER_FLYDSL_FORCE=1 -e AITER_FORCE_GFX1250=0 \
    -e ATOM_USE_UNIFIED_ATTN=1 -e ATOM_FORCE_ATTN_TRITON=1 \
    -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
    "$MAD_IMG" bash /out/server_cmd.sh >"$ARM_OUT/cid.txt" 2>&1 || {
      say "$tag: docker run failed"; return 1; }

  say "$tag: waiting for server"
  local ready=0
  for i in $(seq 1 2400); do
    curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
      say "$tag: container died during load"
      docker logs "$NAME" 2>&1 | grep -iE 'error|notimplemented|traceback' | tail -10 | tee -a "$STATE"
      docker rm -f "$NAME" >/dev/null 2>&1; return 1; }
    (( i % 300 == 0 )) && say "  $tag still loading (${i}s)"
    sleep 1
  done
  [[ $ready -eq 1 ]] || { say "$tag: not ready in 2400s"; docker rm -f "$NAME" >/dev/null 2>&1; return 1; }
  say "$tag: server READY"

  for C in $CONC; do
    timeout 5400 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${port}" \
      --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
      --request-rate inf --random-range-ratio 0.8 --trust-remote-code \
      --max-concurrency "$C" --num-prompts $(( C * 10 )) \
      --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
      --result-dir /out --result-filename "c${C}.json" >"$ARM_OUT/c${C}.log" 2>&1
    if [[ -f "$ARM_OUT/c${C}.json" ]]; then
      say "  $tag c=$C: $($PY -c "import json;d=json.load(open('$ARM_OUT/c${C}.json'));print('%.1f tok/s ttft=%.0fms tpot=%.2fms'%(d['output_throughput'],d['median_ttft_ms'],d['median_tpot_ms']))")"
    else
      say "  $tag c=$C: FAILED (no json)"
    fi
  done

  docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  sleep 10
  say "$tag: done"
}

run_arm tp_only ""                          8020 || say "tp_only arm incomplete"
run_arm ep      "--enable-expert-parallel"  8021 || say "ep arm incomplete"

nA=$(ls "$OUT"/tp_only/c*.json 2>/dev/null | wc -l)
nB=$(ls "$OUT"/ep/c*.json 2>/dev/null | wc -l)
say "collected tp_only=$nA ep=$nB result files"
echo "EPM_TP_N=$nA" >>"$STATE"; echo "EPM_EP_N=$nB" >>"$STATE"
if [[ "$nA" -eq 0 || "$nB" -eq 0 ]]; then
  say "one arm produced nothing — not generating the A/B"
  echo "EPM_STATUS=incomplete" >>"$STATE"; exit 1
fi
echo "EPM_STATUS=ok" >>"$STATE"

MD="$BENCH_ROOT/results/kimi-k3-ep-matched.md"
if [[ ! -f "$MD" ]]; then
  cat >"$MD" <<HDR
# Kimi-K3 — expert parallelism vs TP-only, matched admission cap

Both arms: MAD-pinned image, MAD env vars, \`--max-num-seqs $MAXSEQS\`, TP=8, ISL/OSL $ISL/$OSL.
The **only** difference between them is \`--enable-expert-parallel\`.

This supersedes the A/B in \`kimi-k3-mad.md\` section 6, where the two arms accidentally used
different \`--max-num-seqs\` (64 vs 256) and only the c=64 point was comparable. Here the cap
is identical, so every concurrency point is a valid comparison.

---

## Source data

| What | Where |
|---|---|
| TP-only arm | \`$(basename "$OUT")/tp_only/c<N>.{json,log}\` |
| EP arm | \`$(basename "$OUT")/ep/c<N>.{json,log}\` |
| Driver state | \`$(basename "$OUT")/STATE.txt\` |
HDR
fi

say "generating $MD"
$PY analyze_kimi_ep.py "$OUT/ep" "$OUT/tp_only" "$MD" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $MD"
say "MATCHED EP A/B DONE"
