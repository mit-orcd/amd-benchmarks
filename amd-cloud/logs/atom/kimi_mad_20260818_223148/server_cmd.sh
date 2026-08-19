#!/usr/bin/env bash
set -uo pipefail

# The MAD-pinned image ships WITHOUT flash-linear-attention ('fla'), but ATOM's Kimi-K3
# model file imports it unconditionally for the KDA prefill path
# (kimi_k3.py:749 'from fla.ops.kda import chunk_kda', no flag guard, no fallback).
# 69 of Kimi-K3's 93 layers are KDA linear-attention, so without it the server loads and
# answers /v1/models, then dies on the first real request with ModuleNotFoundError.
# This is a packaging gap in the dated MAD tag -- rocm/atom-dev:latest does ship it.
# See ../notes-kimi-k3.md "Rerun attempt 1" for the full diagnosis.
if ! python -c 'import fla' 2>/dev/null; then
  echo "[fix] installing flash-linear-attention (missing from this image)"
  pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3
  python -c 'from fla.ops.kda import chunk_kda; print("[fix] fla.ops.kda OK")' || {
    echo "[fix] FATAL: fla still unimportable after install"; exit 1; }
else
  echo "[fix] fla already present"
fi

exec python -m atom.entrypoints.openai_server \
  --model /model --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code --max-model-len 16384 \
  --max-num-seqs 64 --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.93 --block-size 128 \
  --server-port 8010 \
  --no-enable_prefix_caching 2>&1 | tee /out/atom_server.log
