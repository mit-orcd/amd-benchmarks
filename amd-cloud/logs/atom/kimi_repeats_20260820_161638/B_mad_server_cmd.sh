#!/usr/bin/env bash
set -uo pipefail
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -2
  python -c 'from fla.ops.kda import chunk_kda' || exit 1
fi
exec python -m atom.entrypoints.openai_server \
  --model /model --tensor-parallel-size 8 \
  --server-port 8015 \
  --kv_cache_dtype fp8 --max-num-seqs 64 \
  --gpu-memory-utilization 0.93 --trust-remote-code \
  --max-model-len 16384 --block-size 128 --no-enable_prefix_caching \
  --max-num-batched-tokens 10240 2>&1 | tee /out/B_mad_server.log
