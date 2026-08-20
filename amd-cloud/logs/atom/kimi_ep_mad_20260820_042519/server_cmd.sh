#!/usr/bin/env bash
set -uo pipefail
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -2
  python -c 'from fla.ops.kda import chunk_kda' || exit 1
fi
exec python -m atom.entrypoints.openai_server \
  --model /model --tensor-parallel-size 8 \
  --server-port 8017 \
  --kv_cache_dtype fp8 --max-num-seqs 256 \
  --gpu-memory-utilization 0.93 --trust-remote-code \
  --max-model-len 16384 --max-num-batched-tokens 10240 \
  --block-size 128 --no-enable_prefix_caching \
  --enable-expert-parallel 2>&1 | tee /out/atom_server.log
