#!/usr/bin/env bash
set -uo pipefail
cd /workspace/Megatron-LM 2>/dev/null || cd /workspace || cd /
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 --nnodes=1 \
  --master_addr=localhost --master_port=29566 \
  pretrain_gpt.py \
    --num-layers 40 \
    --hidden-size 6144 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 48 \
    --group-query-attention --num-query-groups 8 \
    --seq-length 4096 --max-position-embeddings 4096 \
    --vocab-size 50304 \
    --swiglu \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --micro-batch-size 4 --global-batch-size 32 \
    --bf16 \
    --use-distributed-optimizer \
    --ddp-bucket-size 250000000 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --log-throughput \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --train-iters 50 \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine \
    --lr-warmup-iters 5 --lr-decay-iters 30 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --split 949,50,1 \
    --log-interval 1 \
    --no-save-optim --no-load-optim \
    --eval-iters 0 \
    --attention-backend fused
