#!/usr/bin/env bash
set -uo pipefail
exec python -m atom.entrypoints.openai_server \
  --model /model \
  --tensor-parallel-size 8 \
  --server-port 8001 \
  --kv_cache_dtype fp8 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
   2>&1 | tee /out/atom_server.log
