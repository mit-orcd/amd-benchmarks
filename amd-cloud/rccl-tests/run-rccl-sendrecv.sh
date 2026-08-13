#!/usr/bin/env bash
# sendrecv-only rerun. The reference run needed this because sendrecv got killed
# when alltoallv OOMed at N=5; our ALLTOALL_MAX cap should prevent that, but keep
# the script for targeted reruns.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
COLLECTIVES=sendrecv \
OUT_TAG=sendrecv \
exec "$BENCH_ROOT/rccl-tests/run-rccl-all.sh"
