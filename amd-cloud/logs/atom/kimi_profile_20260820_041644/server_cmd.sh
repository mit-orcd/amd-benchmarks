#!/usr/bin/env bash
set -uo pipefail
if ! python -c 'import fla' 2>/dev/null; then pip install --no-cache-dir flash-linear-attention 2>&1|tail -2; fi
exec python -m atom.entrypoints.openai_server \
  --model /model --tensor-parallel-size 8 \
  --server-port 8015 \
  --kv_cache_dtype fp8 --max-num-seqs 256 \
  --gpu-memory-utilization 0.93 --trust-remote-code \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --block-size 128 --no-enable_prefix_caching \
  --torch-profiler-dir /traces \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' 2>&1 | tee /out/atom_server.log
