#!/usr/bin/env bash
set -uo pipefail
# rocm/atom-dev:latest ships flash-linear-attention; guard anyway so this script
# is safe if the tag moves (the MAD-pinned image lacked it -- see notes-kimi-k3.md).
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3
  python -c 'from fla.ops.kda import chunk_kda; print("[fix] fla.ops.kda OK")' || exit 1
else
  echo "[fix] fla already present"
fi
exec python -m atom.entrypoints.openai_server \
  --model /model \
  --tensor-parallel-size 8 \
  --server-port 8012 \
  --kv_cache_dtype fp8 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.93 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' 2>&1 | tee /out/atom_server.log
