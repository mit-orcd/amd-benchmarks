#!/usr/bin/env bash
# Fetch ATOM benchmark checkpoints to $SCRATCH/models. Idempotent and resumable:
# `hf download` skips files it already has, so re-running after an interruption
# continues rather than restarting.
#
# Usage: ./download_models.sh [tier1|tier2|tier3|all]     (default: all)
#
# Safe to run while a GPU benchmark is in flight: this is network + disk I/O to
# /mnt/scratch (a separate NVMe from /), with no GPU involvement. It runs under
# `nice` so it cannot starve a benchmark's host-side threads.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh

WHICH="${1:-all}"
HF="$SCRATCH/venv/bin/hf"
DEST="$SCRATCH/models"
mkdir -p "$DEST"

# tag : HF repo : local dir : approx size
MODELS=(
  "tier1:Qwen/Qwen3-8B-FP8:Qwen3-8B-FP8:8.9 GB"
  "tier2:RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8:Llama-3.1-70B-Instruct-FP8:68 GB"
  "tier3:moonshotai/Kimi-K3:Kimi-K3:1.56 TB"
)

for entry in "${MODELS[@]}"; do
  IFS=: read -r tag repo dir size <<<"$entry"
  [[ "$WHICH" == "all" || "$WHICH" == "$tag" ]] || continue
  target="$DEST/$dir"
  echo "[$(date -Iseconds)] $tag: $repo -> $target ($size)"

  avail=$(df -B1 --output=avail "$DEST" | tail -1)
  echo "  free on $(df --output=target "$DEST" | tail -1): $(numfmt --to=iec "$avail")"

  nice -n 10 "$HF" download "$repo" --local-dir "$target"
  rc=$?
  echo "[$(date -Iseconds)] $tag: rc=$rc size=$(du -sh "$target" 2>/dev/null | cut -f1)"
  if [[ $rc -ne 0 ]]; then
    echo "[$(date -Iseconds)] $tag: FAILED — re-run this script to resume" >&2
  fi
done
echo "[$(date -Iseconds)] downloads done"
df -h "$DEST" | tail -1
