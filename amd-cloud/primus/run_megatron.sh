#!/usr/bin/env bash
# Megatron-LM llama2-7B BF16 pretrain via Primus, N=1..8 (the headline benchmark).
#
# GBS is computed per N instead of fixed at 256: the reference run needed three rerun
# scripts because 256 is not divisible by MBS(4) x DP(N) for N in {3,5,6,7}.
#   GBS(N) = MBS x N x GRAD_ACC = 4 x N x 8 = 32N   (weak scaling, constant work/GPU)
#
# EXP path corrected for this image: the plan named
# examples/megatron/configs/llama2_7B-pretrain.yaml, which does not exist. The image
# (and upstream HEAD) ship per-arch configs; MI355X/llama2_7B-BF16-pretrain.yaml is
# the right one and is arch-tuned for this box.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"          # fallback: rocm/primus:v25.9_gfx950
EXP="${EXP:-examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml}"
RUN_ID="${1:-$(cat "$LOG_ROOT/primus/CURRENT_RUN_ID.txt" 2>/dev/null || date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/sweep-$RUN_ID; mkdir -p "$BASE"
SUM=$BASE/summary.txt
MBS="${MBS:-4}"; GRAD_ACC="${GRAD_ACC:-8}"; TIMEOUT="${TIMEOUT:-3600}"
GPU_COUNTS="${GPU_COUNTS:-1 2 3 4 5 6 7 8}"
assert_gpus_idle

echo "================ MEGATRON $(date -Iseconds) image=$IMG exp=$EXP ================" | tee -a "$SUM"

run_megatron() {
  local N=$1 devs port log start rc dur status GBS
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  GBS=$(( MBS * N * GRAD_ACC ))
  log="$BASE/megatron-llama2_7B-bf16_N${N}.log"; start=$(date +%s)
  echo "----- megatron N=$N GBS=$GBS MBS=$MBS devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout --signal=TERM --kill-after=30s "$TIMEOUT" \
    docker run $(dgpu_args) -w /workspace/Primus \
      -e EXP="$EXP" \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
      "$IMG" bash -c '
        MY=/workspace/Primus/primus/configs/models/megatron/llama2_7B.yaml
        # offline tokenizer: no HF download, mock_data works
        sed -i "s|tokenizer_type: Llama2Tokenizer|tokenizer_type: NullTokenizer|" "$MY"
        sed -i "/^tokenizer_model:/d" "$MY"
        grep -q "^vocab_size:" "$MY" || echo "vocab_size: 32000" >> "$MY"
        bash examples/run_pretrain.sh global_batch_size='"$GBS"' micro_batch_size='"$MBS"'
      ' >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  case $rc in 0) status=OK ;; 124) status="TIMEOUT(${TIMEOUT}s)" ;; *) status="FAIL(rc=$rc)" ;; esac
  echo "  $status duration=${dur}s log=$log" | tee -a "$SUM"
}

for N in $GPU_COUNTS; do run_megatron "$N"; done
echo "[megatron] $(date -Iseconds) DONE" | tee -a "$SUM"
