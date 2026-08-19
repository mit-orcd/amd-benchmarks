#!/usr/bin/env bash
# A/B experiment: Kimi-K3 with expert parallelism ON, against the existing TP-only baseline.
#
# Hypothesis (from results/kimi-k3-base.md section 3/5): with EP off, TP shards every expert
# across all 8 GPUs, so each GPU reads a slice of EVERY activated expert -- ~116 GB/step of
# HBM traffic at c=64 (~29% of HBM bandwidth) while XGMI sits at ~1%. EP instead places
# whole experts on specific ranks: fewer, complete expert reads per GPU, paid for with
# all-to-all token routing over the idle interconnect.
#
# This is an EXPERIMENT, not a correction. The baseline uses the validated recipe flag set
# from ATOM/recipes/Kimi-K3.md; EP is NOT in that validated set, so it may fail outright or
# need an explicit EP size. The baseline stays the reference result either way.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

MKIMI="${MKIMI:-$SCRATCH/models/Kimi-K3}"
CONC="${KIMI_CONC:-1 2 4 8 16 32 64}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
PORT="${PORT:-8003}"

DRV=$LOG_ROOT/atom/kimi_ep_ab_$(date +%Y%m%d_%H%M%S); mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "Kimi-K3 EP A/B start (driver log: $DRV)"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy."; exit 1
fi
if pgrep -af 'atom.entrypoints' >/dev/null 2>&1; then
  say "ABORT: a foreign ATOM server is running."; exit 1
fi
say "GPUs idle, proceeding"

# Same flags as the validated tier-3 recipe, PLUS --enable-expert-parallel.
KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

say "starting server: TP=8 + EP enabled, port=$PORT"
MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.93 READY_TIMEOUT=2400 \
  ./run_atom_server.sh "$MKIMI" 8 "$PORT" \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --block-size 128 \
    --no-enable_prefix_caching \
    --enable-expert-parallel \
    --online_quant_config "'$KIMI_QUANT'" \
    >"$DRV/server.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  say "EP SERVER FAILED TO START (rc=$rc) — this is itself a finding: EP is not in the"
  say "validated recipe set and may be unsupported for the KDA/MLA hybrid."
  tail -25 "$DRV/server.log" | tee -a "$STATE"
  ./stop_atom_server.sh >/dev/null 2>&1
  echo "EP_STATUS=server_failed" >>"$STATE"
  exit 1
fi
say "server up"

./run_atom_bench.sh "$MKIMI" "$PORT" "$ISL" "$OSL" "$CONC" >"$DRV/bench.log" 2>&1
brc=$?
SWEEP=$(cat "$LOG_ROOT/atom/CURRENT_SWEEP_DIR.txt" 2>/dev/null)
say "bench rc=$brc sweep=$SWEEP"
./stop_atom_server.sh >"$DRV/stop.log" 2>&1
say "server stopped"

if [[ $brc -ne 0 ]]; then
  say "EP BENCH FAILED — see $DRV/bench.log"
  tail -8 "$DRV/bench.log" | tee -a "$STATE"
  echo "EP_STATUS=bench_failed" >>"$STATE"
  exit 1
fi

echo "EP_STATUS=ok" >>"$STATE"
echo "EP_SWEEP=$SWEEP" >>"$STATE"

say "appending analysis to results/kimi-k3-base.md"
$PY analyze_kimi_ep.py "$SWEEP" \
   "$LOG_ROOT/atom/sweep_20260814_164903" \
   "$BENCH_ROOT/results/kimi-k3-base.md" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/kimi-k3-base.md"
say "EP A/B DONE"
