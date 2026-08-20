#!/usr/bin/env bash
set -uo pipefail
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -2
  python -c 'from fla.ops.kda import chunk_kda' || exit 1
fi
exec python -m atom.entrypoints.openai_server \
  --model /model --tensor-parallel-size 8 \
  --server-port 8014 \
  --kv_cache_dtype fp8 --max-num-seqs 64 \
  --gpu-memory-utilization 0.93 --trust-remote-code \
  --max-model-len 16384 --block-size 128 --no-enable_prefix_caching \
  --max-num-batched-tokens 16384 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' 2>&1 | tee /out/A_original_server.log
