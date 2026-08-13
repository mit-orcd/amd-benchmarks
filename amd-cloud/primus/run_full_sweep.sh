#!/usr/bin/env bash
# Primus microbench sweep: gemm, gemm-dense, gemm-deepseek, attention, rccl x N=1..8 (~1 h).
# Megatron gets its own script so a Megatron failure doesn't cost the whole hour.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
IMG="${IMG:-rocm/primus:v26.5}"
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
BASE=$LOG_ROOT/primus/sweep-$RUN_ID; mkdir -p "$BASE"
OUT_HOST=$BENCH_ROOT/primus/sweep_out_$RUN_ID; mkdir -p "$OUT_HOST"
OUT_CTR=/out
SUM=$BASE/summary.txt
assert_gpus_idle

{ echo "Primus full sweep $RUN_ID"; echo "Image      : $IMG";
  echo "Driver log : $BASE"; echo "Bench out  : $OUT_HOST";
  echo "Started    : $(date -Iseconds)"; echo; } | tee "$SUM"

run() {
  local name=$1 N=$2 t=$3 cmd=$4 devs port log start rc dur status
  devs=$(seq -s, 0 $((N-1))); port=$((29500 + RANDOM % 500 + N))
  log="$BASE/${name}_N${N}.log"; start=$(date +%s)
  echo "----- $name N=$N port=$port devs=$devs $(date -Iseconds) -----" | tee -a "$SUM"
  timeout --signal=TERM --kill-after=30s "$t" \
    docker run $(dgpu_args) -v "$OUT_HOST":/out -w /workspace/Primus \
      -e HIP_VISIBLE_DEVICES="$devs" -e ROCR_VISIBLE_DEVICES="$devs" \
      -e GPUS_PER_NODE="$N" -e NNODES=1 -e NODE_RANK=0 \
      -e MASTER_ADDR=localhost -e MASTER_PORT="$port" \
      -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
      "$IMG" bash -c "$cmd" >"$log" 2>&1
  rc=$?; dur=$(($(date +%s)-start))
  case $rc in 0) status=OK ;; 124) status="TIMEOUT(${t}s)" ;; *) status="FAIL(rc=$rc)" ;; esac
  echo "  $status duration=${dur}s log=$log" | tee -a "$SUM"
}

for N in 1 2 3 4 5 6 7 8; do
  echo "================ N=$N ================" | tee -a "$SUM"
  run gemm          $N  300 "./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10 --output-file $OUT_CTR/gemm_N${N}.md"
  run gemm-dense    $N  600 "./primus-cli direct -- benchmark gemm-dense --duration 5 --output-file $OUT_CTR/gemm-dense_N${N}.md"
  run gemm-deepseek $N  600 "./primus-cli direct -- benchmark gemm-deepseek --duration 5 --output-file $OUT_CTR/gemm-deepseek_N${N}.md"
  run attention     $N 1200 "./primus-cli direct -- benchmark attention --backend flash --mbs-list 4 --report-csv-path $OUT_CTR/attention_N${N}.csv"
  if (( N >= 2 )); then
    # NB: no --op flag -- argparse advertises all_reduce but the backend wants allreduce;
    # omitting it uses the correct default.
    run rccl        $N  900 "./primus-cli direct -- benchmark rccl --output-file $OUT_CTR/rccl_N${N}.md"
  else
    echo "----- rccl N=1 SKIPPED (collective needs N>=2) -----" | tee -a "$SUM"
  fi
done

echo; echo "Finished   : $(date -Iseconds)" | tee -a "$SUM"
