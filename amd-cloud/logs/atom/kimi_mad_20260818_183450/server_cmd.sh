#!/usr/bin/env bash
set -uo pipefail
exec python -m atom.entrypoints.openai_server \
  --model /model --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code --max-model-len 16384 \
  --max-num-seqs 64 --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.93 --block-size 128 \
  --server-port 8010 \
  --no-enable_prefix_caching 2>&1 | tee /out/atom_server.log
