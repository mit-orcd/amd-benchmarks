#!/usr/bin/env bash
# Megatron-LM BF16 throughput sweep on AMD MI355X (single node)
# Container: megatron-lm.sif (Singularity, ROCm/Megatron-LM image)
# Goal:      report TFLOP/s/GPU at N_GPUS = 1..8 (weak scaling — GBS scales
#            with N_GPUS so per-GPU micro-batch is constant). This isolates
#            per-GPU throughput and the DP collective overhead as N grows.
#
# Run:       bash /home/v89592/shaohao/megatron-lm/work/run.sh
set -euo pipefail

# ---------------------------------------------------------------- paths --
ROOT=/home/v89592/shaohao/megatron-lm
SIF="$ROOT/megatron-lm.sif"
MEGATRON_DIR="$ROOT/Megatron-LM"      # ROCm fork, rocm_dev branch (bind-mounted)
WORK_DIR="$ROOT/work"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="$LOG_DIR/sweep_${STAMP}"
mkdir -p "$SWEEP_DIR"
SUMMARY_FILE="$SWEEP_DIR/sweep_summary.txt"

[[ -f "$SIF" ]]          || { echo "missing image: $SIF"          >&2; exit 1; }
[[ -d "$MEGATRON_DIR" ]] || { echo "missing source: $MEGATRON_DIR" >&2; exit 1; }

# ----------------------------------------------------------- topology ---
N_NODES=1
GPU_COUNTS=(1 2 3 4 5 6 7 8)

# Model shape — tuned for MI355X (288 GB HBM3e, gfx950, BF16 ~5 PFLOPS peak).
# Pure DP (TP=PP=1) keeps GEMMs local and avoids cross-GPU AllReduce in fwd/bwd;
# distributed optimizer shards Adam state across DP ranks so the full model
# still fits comfortably per GPU without activation recompute.
NUM_LAYERS=40
HIDDEN=6144
FFN=16384
NUM_HEADS=48
NUM_KV_HEADS=8                        # GQA — same head dim, fewer KV projections
SEQ_LEN=4096
MAX_POS=$SEQ_LEN

MICRO_BS=2                            # held constant across the sweep
# GBS is set per-iteration: GBS = MICRO_BS * N_GPUS  (weak scaling).

TRAIN_ITERS=50
LOG_INTERVAL=5

# ----------------------------------------------------------- RCCL / xGMI -
# Single-node interconnect = AMD Infinity Fabric (xGMI) between GPUs.
# Disable IB (no fabric), let RCCL use xGMI peer-to-peer + SHM, enable MSCCL
# for tuned all-reduce paths on AMD.
#
# HIP_VISIBLE_DEVICES is *not* baked into this block — it is set per-run
# inside the sweep loop so each iteration sees only N_GPUS devices.
read -r -d '' CONTAINER_ENV <<'EOF' || true
ROCM_PATH=/opt/rocm
HSA_NO_SCRATCH_RECLAIM=1
PYTORCH_ROCM_ARCH=gfx950
# PyTorch in megatron-lm.sif is built without gfx950 code objects
# (torch.cuda.get_arch_list() == []), so HIP init fails on MI355X with
# "No HIP GPUs are available". Alias gfx950 -> gfx942 so the existing
# MI300X kernels run on MI355X (binary-compatible at gfx9 ISA level).
# Remove this once a gfx950-native PyTorch wheel is in the image.
HSA_OVERRIDE_GFX_VERSION=9.4.2

# RCCL / NCCL
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
RCCL_MSCCL_ENABLE=1
NCCL_PROTO=Simple,LL,LL128
NCCL_ALGO=Ring,Tree
NCCL_DEBUG=WARN

# Torch / perf
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
TORCH_NCCL_AVOID_RECORD_STREAMS=1
EOF

BASE_SENV_ARGS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  BASE_SENV_ARGS+=(--env "$line")
done <<< "$CONTAINER_ENV"

# ----------------------------------------------------------- Megatron CLI
MODEL_ARGS=(
  --num-layers              "$NUM_LAYERS"
  --hidden-size             "$HIDDEN"
  --ffn-hidden-size         "$FFN"
  --num-attention-heads     "$NUM_HEADS"
  --group-query-attention
  --num-query-groups        "$NUM_KV_HEADS"
  --seq-length              "$SEQ_LEN"
  --max-position-embeddings "$MAX_POS"
  --position-embedding-type rope
  --swiglu
  --normalization           RMSNorm
  --untie-embeddings-and-output-weights
)

# ----------------------------------------------------------- sweep ------
echo "==== Megatron-LM BF16 sweep ($(date)) ===="
echo "image  : $SIF"
echo "source : $MEGATRON_DIR"
echo "shape  : L=$NUM_LAYERS H=$HIDDEN FFN=$FFN heads=$NUM_HEADS kv=$NUM_KV_HEADS seq=$SEQ_LEN"
echo "micro  : $MICRO_BS  (GBS = MICRO_BS * N_GPUS, weak scaling)"
echo "sweep  : N_GPUS in {${GPU_COUNTS[*]}}"
echo "outdir : $SWEEP_DIR"
echo

# Header of the sweep summary table.
printf '%-7s %-6s %-13s %-13s %s\n' \
  "N_GPUS" "GBS" "last_TF/GPU" "best_TF/GPU" "log" | tee "$SUMMARY_FILE"

for N_GPUS in "${GPU_COUNTS[@]}"; do
  GBS=$(( MICRO_BS * N_GPUS ))
  DEVS=$(seq -s, 0 $(( N_GPUS - 1 )))
  LOG_FILE="$SWEEP_DIR/bench_bf16_n${N_GPUS}.log"

  SENV_ARGS=("${BASE_SENV_ARGS[@]}" --env "HIP_VISIBLE_DEVICES=$DEVS")

  TRAIN_ARGS=(
    --mock-data
    --tokenizer-type          NullTokenizer
    --vocab-size              50304

    --tensor-model-parallel-size       1
    --pipeline-model-parallel-size     1
    --data-parallel-sharding-strategy  no_shard
    --use-distributed-optimizer

    --micro-batch-size        "$MICRO_BS"
    --global-batch-size       "$GBS"

    --train-iters             "$TRAIN_ITERS"
    --lr                      3e-4
    --min-lr                  3e-5
    --lr-decay-style          cosine
    --lr-warmup-iters         5
    --lr-decay-iters          30
    --weight-decay            0.1
    --adam-beta1              0.9
    --adam-beta2              0.95
    --clip-grad               1.0

    --bf16
    --use-flash-attn

    --eval-interval           1000000
    --save-interval           1000000
    --log-interval            "$LOG_INTERVAL"
    --log-throughput
    --timing-log-level        2
    --timing-log-option       all
  )

  echo "---- N_GPUS=$N_GPUS  GBS=$GBS  devices=$DEVS  -> $LOG_FILE"

  # Build the command that runs inside the container. Using bash -lc so the
  # container's PATH / torchrun resolve correctly.
  INNER_CMD=$(cat <<EOF
set -euo pipefail
cd "$MEGATRON_DIR"
export PYTHONPATH="$MEGATRON_DIR:\${PYTHONPATH:-}"
echo "[in-container] python: \$(python3 --version 2>&1)"
echo "[in-container] torch : \$(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"
echo "[in-container] HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-unset}"
echo
torchrun \
  --standalone \
  --nnodes=$N_NODES \
  --nproc_per_node=$N_GPUS \
  "$MEGATRON_DIR/pretrain_gpt.py" \
  ${MODEL_ARGS[@]} \
  ${TRAIN_ARGS[@]}
EOF
)

  # Don't let one failing N_GPUS abort the rest of the sweep.
  set +e
  singularity exec \
    --rocm \
    --bind "$ROOT:$ROOT" \
    "${SENV_ARGS[@]}" \
    "$SIF" \
    bash -lc "$INNER_CMD" 2>&1 | tee "$LOG_FILE"
  RC=${PIPESTATUS[0]}
  set -e

  # Pull per-run throughput from the log.
  if grep -q "TFLOP/s/GPU" "$LOG_FILE"; then
    mapfile -t VALS < <(grep -oE "throughput per GPU \(TFLOP/s/GPU\): *[0-9]+\.[0-9]+" "$LOG_FILE" \
                         | grep -oE "[0-9]+\.[0-9]+")
    LAST="${VALS[-1]}"
    BEST="$(printf '%s\n' "${VALS[@]}" | sort -g | tail -1)"
  else
    LAST="ERR"
    BEST="ERR"
  fi

  printf '%-7s %-6s %-13s %-13s %s\n' \
    "$N_GPUS" "$GBS" "$LAST" "$BEST" "$LOG_FILE" | tee -a "$SUMMARY_FILE"

  if [[ "$RC" -ne 0 ]]; then
    echo "[!] N_GPUS=$N_GPUS exited with status $RC — continuing sweep"
  fi
done

echo
echo "==== sweep complete ===="
echo "summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
