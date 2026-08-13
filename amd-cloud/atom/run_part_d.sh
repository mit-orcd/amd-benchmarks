#!/usr/bin/env bash
# PART D driver: gate -> tier-1 (8B, TP1) -> tier-2 (70B, TP8) -> analysis.
#
# Strictly sequential, one server at a time. Designed to run under nohup and survive
# logout, like run_part_a.sh. Every stage refuses rather than forces: if the GPUs are
# busy or a foreign ATOM server is up, it stops instead of stomping.
#
# Usage: ./run_part_d.sh [tier1|tier2|all]     (default: all)
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
cd "$(dirname "$0")"

WHICH="${1:-all}"
IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
M8B="${M8B:-$SCRATCH/models/Qwen3-8B-FP8}"
M70B="${M70B:-$SCRATCH/models/Llama-3.1-70B-Instruct-FP8}"
MKIMI="${MKIMI:-$SCRATCH/models/Kimi-K3}"
CONC="${CONC:-1 2 4 8 16 32 64 128 256}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"

# Kimi-K3 launch flags, taken verbatim from ATOM/recipes/Kimi-K3.md. These are not
# optional tuning knobs:
#   -tp 8            : required for the 1.56 TB model to fit at all
#   gpu-mem-util .93 : so the CUDA-graph pool fits beside the KDA per-request state cache
#   no prefix caching: the KDA recurrent state is per-request and cannot be rebuilt from
#                      the paged MLA cache, so prefix reuse would be incorrect
#   online_quant_config: PTPC-FP8 for attention/dense; the routed MoE experts are already
#                      MXFP4 in the checkpoint and are excluded here
KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
KIMI_ARGS=(
  --max-model-len 16384
  --max-num-batched-tokens 16384
  --block-size 128
  --no-enable_prefix_caching
  --online_quant_config "'$KIMI_QUANT'"
)

TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/atom/part_d_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

say "PART D start (which=$WHICH, driver log: $DRV)"

# ---- guard ---------------------------------------------------------------------
busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy — another benchmark is running. Part D must not overlap A-C."
  exit 1
fi
if pgrep -af 'atom.entrypoints' >/dev/null 2>&1; then
  say "ABORT: a foreign ATOM server is already running on this host."
  exit 1
fi
say "GPUs idle, no foreign ATOM server"

# ---- stage 0: image gate --------------------------------------------------------
say "STAGE 0 gfx950 gate"
docker run --rm $(dgpu_args) "$IMG" \
  python -c "import torch;print(torch.__version__);print(torch.cuda.get_arch_list());print('devices',torch.cuda.device_count())" \
  >"$DRV/gate.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]] || ! grep -q gfx950 "$DRV/gate.log"; then
  say "ABORT: gate failed (rc=$rc) or gfx950 missing from arch list — see $DRV/gate.log"
  tail -5 "$DRV/gate.log" | tee -a "$STATE"
  exit 1
fi
say "gate OK: $(grep -o 'devices [0-9]*' "$DRV/gate.log")"

run_tier() {
  local tag=$1 model=$2 tp=$3 port=$4 conc=$5; shift 5
  local extra=("$@")
  if [[ ! -d "$model" ]]; then
    say "SKIP $tag: model dir missing ($model)"
    return 0
  fi
  say "----- $tag: $(basename "$model") TP=$tp port=$port -----"
  ./run_atom_server.sh "$model" "$tp" "$port" "${extra[@]}" >"$DRV/${tag}_server.log" 2>&1
  if [[ $? -ne 0 ]]; then
    say "$tag: server failed to start — see $DRV/${tag}_server.log"
    tail -20 "$DRV/${tag}_server.log" | tee -a "$STATE"
    ./stop_atom_server.sh >/dev/null 2>&1
    return 1
  fi
  say "$tag: server up"
  ./run_atom_bench.sh "$model" "$port" "$ISL" "$OSL" "$conc" >"$DRV/${tag}_bench.log" 2>&1
  say "$tag: bench rc=$? sweep=$(cat "$LOG_ROOT/atom/CURRENT_SWEEP_DIR.txt" 2>/dev/null)"
  ./stop_atom_server.sh >"$DRV/${tag}_stop.log" 2>&1
  say "$tag: server stopped"
  sleep 10   # let VRAM drain before the next tier
}

[[ "$WHICH" == "all" || "$WHICH" == "tier1" ]] && run_tier tier1 "$M8B"  1 8000 "$CONC"
[[ "$WHICH" == "all" || "$WHICH" == "tier2" ]] && run_tier tier2 "$M70B" 8 8001 "$CONC"
# Kimi-K3: max-num-seqs 64 per the recipe, so the concurrency list is capped there --
# driving past max-num-seqs measures queueing, not the engine.
if [[ "$WHICH" == "all" || "$WHICH" == "tier3" ]]; then
  MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.93 READY_TIMEOUT=2400 \
    run_tier tier3 "$MKIMI" 8 8002 "${KIMI_CONC:-1 2 4 8 16 32 64}" "${KIMI_ARGS[@]}"
fi

# ---- analysis -------------------------------------------------------------------
say "STAGE analysis"
$PY analyze_atom.py "$LOG_ROOT"/atom/sweep_* -o "$BENCH_ROOT/results" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $BENCH_ROOT/results/atom.{md,csv}"
say "PART D DONE"
