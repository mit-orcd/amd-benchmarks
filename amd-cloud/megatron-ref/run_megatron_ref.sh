#!/usr/bin/env bash
# Apples-to-apples reproduction of the Dell Cloud "GPT-15.6B" Megatron-LM run, so that the
# AMD Cloud MI355X can be placed in the SAME table as B200 (986.0) and Dell MI355X (790.4).
#
# Why this is NOT the Part C Primus run: that one is llama2-7B through Primus with
# primus-turbo fused kernels. The B200 comparison in dell-cloud/megatron-lm/summary.md
# section "vs. NVIDIA B200" used a completely different path:
#   - model  : GPT-15.6B  (L=40, H=6144, FFN=16384, heads=48, GQA kv=8, seq=4096, vocab=50304)
#   - image  : ROCm/Megatron-LM (rocm/megatron-lm), NOT Primus, NO primus-turbo
#   - config : MBS=4, GBS=32 (=MBS x 8), BF16, NO recompute, TP=PP=1, dist-optimizer
#   - tuned  : --ddp-bucket-size 250000000  (this is what produced 790.4, +2.0% over 775.1)
# Every one of those is reproduced below, read out of Dell's own resolved-argument dump in
# logs/tflops_v26.1_tune_20260604_104624/bench_bf16_ddp_bucket_250M.log.
#
# 2026-08-20 CORRECTION. The first completed run (run_20260820_074532) claimed to be
# apples-to-apples but was NOT: eight of Dell's flags were missing, so it built a different
# model and ran it with less comm overlap. Diffed against Dell's resolved-arg dump:
#     swiglu               False  vs True    <- different FFN, ~1.29x the MLP FLOPs
#     normalization    LayerNorm  vs RMSNorm
#     position_embedding  learned  vs rope
#     disable_bias_linear  False  vs True
#     untie_embeddings     False  vs True
#     masked_softmax_fusion True  vs False
#     overlap_grad_reduce  False  vs True    <- grad all-reduce not overlapped w/ backward
#     overlap_param_gather False  vs True
# That run took 6,266 ms/iter against Dell's 2,114 ms/iter -- ~3x slower on a *cheaper*
# model, so it cannot be reported next to the 790.4 figure. All eight flags are now set.
# It also lacked --log-throughput (log_throughput=False), which is why the driver parsed
# no TFLOP/s and refused to update the report -- correct behaviour, right outcome.
# --eval-iters 0 is the one deliberate deviation from Dell: the post-training eval phase
# crashed in AITER's gfx942 asm FMHA kernel, and eval is not part of the measurement.
#
# HSA_OVERRIDE_GFX_VERSION=9.4.2 IS set below, matching Dell. This was tried unset first
# (2026-08-14 21:10 run) on the premise that a native-gfx950 image wouldn't need it -- that
# premise held for ATOM's rocm/atom-dev (hipBLASLt has its own gfx950 tuning independent of
# torch's compiled arch list) but does NOT hold for this image. Without the override:
#   torch.cuda.get_arch_list() == []   (no compiled code objects at all, confirmed by probe)
#   forward pass ran; backward failed in TE's layernorm_linear wgrad_gemm ->
#     RuntimeError: Unable to find any suitable algorithms  (all 8 ranks, same point)
# Setting the override lets TE/hipBLASLt fall back to its gfx942 kernel set, same as Dell's
# run. This makes the run apples-to-apples with Dell on software stack too, not just config
# -- both sides now execute the same code objects. See megatron-ref/*.log for the failed run.
#
# Usage: ./run_megatron_ref.sh [N_GPUS]     (default 8 — the only N with a B200 reference)
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh

N="${1:-8}"
IMG="${MEGATRON_IMG:-rocm/megatron-lm:v26.1}"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/megatron-ref/run_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

# Dell's exact shape and hyperparameters.
NUM_LAYERS=40; HIDDEN=6144; FFN=16384; NUM_HEADS=48; NUM_KV_HEADS=8
SEQ_LEN=4096; VOCAB=50304
MICRO_BS=4; GBS=$(( MICRO_BS * N ))
# Bisect knobs for the 2026-08-20 17:00 SIGSEGV (crashed on the FIRST training step with the
# full Dell flag set; the model itself built correctly -- 16,223,016,960 params, matching Dell
# exactly). OVERLAP=0 drops --overlap-grad-reduce/--overlap-param-gather, which are pure comm
# optimizations: the model stays byte-for-byte Dell-comparable, only the DP overlap differs.
# GAF=0 additionally drops gradient-accumulation-fusion. Try OVERLAP=0 first.
OVERLAP="${OVERLAP:-1}"
GAF="${GAF:-1}"
TRAIN_ITERS=50
DDP_BUCKET=250000000

say "Megatron-LM reference reproduction (Dell-matched) start"
say "image=$IMG N=$N MBS=$MICRO_BS GBS=$GBS seq=$SEQ_LEN L=$NUM_LAYERS H=$HIDDEN OVERLAP=$OVERLAP GAF=$GAF"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  say "ABORT: $busy GPU(s) busy."; exit 1
fi
say "GPUs idle"

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  say "pulling $IMG (this is a large image; / has $(df -h / | awk 'NR==2{print $4}') free)"
  docker pull "$IMG" >"$OUT/pull.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "ABORT: pull failed (rc=$rc). Tag may not exist — check with:"
    say "  curl -s https://registry.hub.docker.com/v2/repositories/rocm/megatron-lm/tags | head"
    tail -15 "$OUT/pull.log" | tee -a "$STATE"
    exit 1
  fi
  say "pull OK"
fi

devs=$(seq -s, 0 $((N-1)))
LOG="$OUT/megatron_ref_N${N}.log"

# Written to a file rather than inlined, to keep quoting out of docker+bash -c.
OPT_FLAGS=""
[[ "$OVERLAP" == "1" ]] && OPT_FLAGS="$OPT_FLAGS --overlap-grad-reduce --overlap-param-gather"
[[ "$GAF" == "0" ]] && OPT_FLAGS="$OPT_FLAGS --no-gradient-accumulation-fusion"
say "optional flags:${OPT_FLAGS:- (none)}"

CMD="$OUT/cmd.sh"
cat >"$CMD" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/Megatron-LM 2>/dev/null || cd /workspace || cd /
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=$N --nnodes=1 \\
  --master_addr=localhost --master_port=$((29500 + RANDOM % 400)) \\
  pretrain_gpt.py \\
    --num-layers $NUM_LAYERS \\
    --hidden-size $HIDDEN \\
    --ffn-hidden-size $FFN \\
    --num-attention-heads $NUM_HEADS \\
    --group-query-attention --num-query-groups $NUM_KV_HEADS \\
    --seq-length $SEQ_LEN --max-position-embeddings $SEQ_LEN \\
    --vocab-size $VOCAB \\
    --swiglu \\
    --normalization RMSNorm \\
    --position-embedding-type rope \\
    --disable-bias-linear \\
    --untie-embeddings-and-output-weights \\
    --no-masked-softmax-fusion \\
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \\
    --micro-batch-size $MICRO_BS --global-batch-size $GBS \\
    --bf16 \\
    --use-distributed-optimizer \\
    --ddp-bucket-size $DDP_BUCKET \\
    --log-throughput \\
    $OPT_FLAGS \\
    --use-flash-attn \\
    --transformer-impl transformer_engine \\
    --train-iters $TRAIN_ITERS \\
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine \\
    --lr-warmup-iters 5 --lr-decay-iters 30 \\
    --mock-data \\
    --tokenizer-type NullTokenizer \\
    --split 949,50,1 \\
    --log-interval 1 \\
    --no-save-optim --no-load-optim \\
    --eval-iters 0 \\
    --attention-backend fused
EOF
chmod +x "$CMD"

say "launching (timeout 3600s)"
timeout --signal=TERM --kill-after=30s 3600 \
  docker run --rm $(dgpu_args) \
    -v "$OUT":/out \
    -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
    -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
    "$IMG" bash /out/cmd.sh >"$LOG" 2>&1
rc=$?
say "run rc=$rc log=$LOG"

tf=$(grep -oE 'throughput per GPU \(TFLOP/s/GPU\): *[0-9.]+|TFLOP/s/GPU\): *[0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
say "parsed TF/s/GPU = ${tf:-none}"
echo "N=$N" >>"$STATE"; echo "TFLOPS=${tf:-}" >>"$STATE"; echo "RC=$rc" >>"$STATE"

if [[ $rc -ne 0 || -z "$tf" ]]; then
  say "FAILED or no throughput parsed — not updating the report."
  say "last 30 lines:"; tail -30 "$LOG" | tee -a "$STATE"
  exit 1
fi

say "updating PRIMUS_REPORT.md section 1.2"
$PY "$BENCH_ROOT/megatron-ref/update_b200_table.py" "$STATE" \
    "$BENCH_ROOT/results/PRIMUS_REPORT.md" >"$OUT/update.log" 2>&1
say "update rc=$? -> results/PRIMUS_REPORT.md"
say "DONE"
