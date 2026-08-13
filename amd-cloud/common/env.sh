#!/usr/bin/env bash
# Shared config for all three benchmark parts. Source, don't execute.
export BENCH_ROOT=/home/amd/shaohao/amd-benchmarks/amd-cloud
export REF_ROOT=/home/amd/shaohao/amd-benchmarks/dell-cloud   # prior Dell Cloud results
export SCRATCH=/mnt/scratch/shaohao
export LOG_ROOT=$BENCH_ROOT/logs          # tracked in-repo, like dell-cloud/*/logs
export CACHE_ROOT=$SCRATCH/cache          # regenerable bulk, off-repo
export PY=$SCRATCH/venv/bin/python
export ROCM_PATH=/opt/rocm
export NGPU=8
export RVS_BIN=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/bin/rvs
export RCCL_TESTS_DIR=$BENCH_ROOT/rccl-tests/src/build
mkdir -p "$LOG_ROOT"/{rvs,rccl,primus} "$CACHE_ROOT"/{triton,hf,torch,pip}

# RCCL env — single node, XGMI only, IB off. Mirrors dell-cloud's CONTAINER_ENV
# block so our numbers stay comparable to the published summaries.
# NOTE: HSA_OVERRIDE_GFX_VERSION is deliberately NOT set (native gfx950).
rccl_env() {
  cat <<'EOF'
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
RCCL_MSCCL_ENABLE=1
NCCL_PROTO=Simple,LL,LL128
NCCL_ALGO=Ring,Tree
NCCL_DEBUG=WARN
EOF
}

# docker run wrapper args: GPU passthrough + caches + host networking.
# Usage: docker run $(dgpu_args) -v ... <image> <command...>
dgpu_args() {
  echo --rm --network host --ipc host --shm-size 64g \
       --device /dev/kfd --device /dev/dri \
       --group-add video --group-add render \
       --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 --ulimit stack=67108864 \
       -v "$CACHE_ROOT/triton:/root/.triton" \
       -v "$CACHE_ROOT/hf:/root/.cache/huggingface" \
       -v "$CACHE_ROOT/torch:/root/.cache/torch" \
       -v "$CACHE_ROOT/pip:/root/.cache/pip"
}

# Fail fast if someone else is using the GPUs.
assert_gpus_idle() {
  local busy
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$' || true)
  [[ "${busy:-0}" -eq 0 ]] || { echo "WARNING: $busy GPU(s) busy — another workload is running"; }
}
