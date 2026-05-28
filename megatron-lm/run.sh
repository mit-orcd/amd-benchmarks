#!/usr/bin/env bash
# Megatron-LM BF16 throughput benchmark on 8x MI355X (single node)
# Container: megatron-lm.sif (Singularity, ROCm/Megatron-LM image)
# Goal:      maximize and report TFLOP/s/GPU via --log-throughput
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
LOG_FILE="$LOG_DIR/bench_bf16_${STAMP}.log"

[[ -f "$SIF" ]]          || { echo "missing image: $SIF"          >&2; exit 1; }
[[ -d "$MEGATRON_DIR" ]] || { echo "missing source: $MEGATRON_DIR" >&2; exit 1; }

# ----------------------------------------------------------- topology ---
N_NODES=1
N_GPUS=8

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

MICRO_BS=2
GBS=$(( MICRO_BS * N_GPUS ))          # one micro-batch per GPU, no grad accum

TRAIN_ITERS=50
LOG_INTERVAL=5

# ----------------------------------------------------------- RCCL / xGMI -
# Single-node interconnect = AMD Infinity Fabric (xGMI) between GPUs.
# Disable IB (no fabric), let RCCL use xGMI peer-to-peer + SHM, enable MSCCL
# for tuned all-reduce paths on AMD.
read -r -d '' CONTAINER_ENV <<'EOF' || true
ROCM_PATH=/opt/rocm
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HSA_NO_SCRATCH_RECLAIM=1
PYTORCH_ROCM_ARCH=gfx950

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

SENV_ARGS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  SENV_ARGS+=(--env "$line")
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

# ----------------------------------------------------------- launch -----
echo "==== Megatron-LM BF16 bench ($(date)) ===="
echo "image : $SIF"
echo "source: $MEGATRON_DIR"
echo "shape : L=$NUM_LAYERS H=$HIDDEN FFN=$FFN heads=$NUM_HEADS kv=$NUM_KV_HEADS seq=$SEQ_LEN"
echo "batch : micro=$MICRO_BS global=$GBS  ($N_NODES node x $N_GPUS GPU)"
echo "log   : $LOG_FILE"
echo

# Build the command that runs inside the container.  Using bash -lc so the
# container's PATH/torchrun resolve correctly.
INNER_CMD=$(cat <<EOF
set -euo pipefail
cd "$MEGATRON_DIR"
export PYTHONPATH="$MEGATRON_DIR:\${PYTHONPATH:-}"
echo "[in-container] python: \$(python3 --version 2>&1)"
echo "[in-container] torch : \$(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"
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

singularity exec \
  --rocm \
  --bind "$ROOT:$ROOT" \
  "${SENV_ARGS[@]}" \
  "$SIF" \
  bash -lc "$INNER_CMD" 2>&1 | tee "$LOG_FILE"

RC=${PIPESTATUS[0]}

# ----------------------------------------------------------- report ----
echo
echo "==== throughput summary ===="
if grep -q "TFLOP/s/GPU" "$LOG_FILE"; then
  # Pull every reported value, then report last + best.
  mapfile -t VALS < <(grep -oE "throughput per GPU \(TFLOP/s/GPU\): *[0-9]+\.[0-9]+" "$LOG_FILE" \
                       | grep -oE "[0-9]+\.[0-9]+")
  printf 'samples : %d\n' "${#VALS[@]}"
  printf 'last    : %s TFLOP/s/GPU\n' "${VALS[-1]}"
  printf 'best    : %s TFLOP/s/GPU\n' \
    "$(printf '%s\n' "${VALS[@]}" | sort -g | tail -1)"
else
  echo "no TFLOP/s/GPU lines found in $LOG_FILE — check the log for errors"
fi

exit "$RC"
