#!/bin/bash
# Full Primus sweep across 1..8 GPUs on MI355X.
# Benches: gemm, gemm-dense, gemm-deepseek, attention, rccl (N>=2),
#          megatron llama2_7B BF16 pretrain (weak-scaling, GBS = 4*N).
set -uo pipefail

PRIMUS_DIR=/home/v89592/shaohao/primus/Primus
SIF=/home/v89592/shaohao/primus/image/primus-v26.3.sif
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
BASE=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
OUT_IN_REPO=${PRIMUS_DIR}/sweep_out_${RUN_ID}
OUT_IN_CTR=/workspace/sweep_out_${RUN_ID}
SUMMARY="${BASE}/summary.txt"

mkdir -p "$BASE" "$OUT_IN_REPO"

echo "Primus full sweep ${RUN_ID}"           | tee "$SUMMARY"
echo "SIF        : $SIF"                     | tee -a "$SUMMARY"
echo "Repo       : $PRIMUS_DIR"              | tee -a "$SUMMARY"
echo "Driver log : $BASE"                    | tee -a "$SUMMARY"
echo "Bench out  : $OUT_IN_REPO"             | tee -a "$SUMMARY"
echo "Started    : $(date -Iseconds)"        | tee -a "$SUMMARY"
echo                                          | tee -a "$SUMMARY"

run() {
    local name=$1 N=$2 timeout_s=$3 cmd=$4
    local devs port log start rc dur status
    devs=$(seq -s, 0 $((N-1)))
    port=$((29500 + (RANDOM % 500) + N))
    log="${BASE}/${name}_N${N}.log"
    start=$(date +%s)

    echo "----- ${name} N=${N} port=${port} devs=${devs} $(date -Iseconds) -----" \
        | tee -a "$SUMMARY"

    timeout --signal=TERM --kill-after=30s "${timeout_s}" \
        singularity exec --rocm --writable-tmpfs \
            --pwd /workspace \
            --bind "${PRIMUS_DIR}":/workspace \
            --env HIP_VISIBLE_DEVICES="${devs}" \
            --env ROCR_VISIBLE_DEVICES="${devs}" \
            --env GPUS_PER_NODE="${N}" \
            --env NNODES=1 \
            --env NODE_RANK=0 \
            --env MASTER_ADDR=localhost \
            --env MASTER_PORT="${port}" \
            "${SIF}" \
            bash -c "${cmd}" \
        > "${log}" 2>&1
    rc=$?
    dur=$(($(date +%s) - start))
    if   [[ $rc -eq 0   ]]; then status="OK"
    elif [[ $rc -eq 124 ]]; then status="TIMEOUT(${timeout_s}s)"
    else                         status="FAIL(rc=${rc})"
    fi
    echo "  ${status}  duration=${dur}s  log=${log}" | tee -a "$SUMMARY"
}

for N in 1 2 3 4 5 6 7 8; do
    echo "================ N=${N} ================" | tee -a "$SUMMARY"

    run gemm $N 300 \
        "./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10 --output-file ${OUT_IN_CTR}/gemm_N${N}.md"

    run gemm-dense $N 600 \
        "./primus-cli direct -- benchmark gemm-dense --duration 5 --output-file ${OUT_IN_CTR}/gemm-dense_N${N}.md"

    run gemm-deepseek $N 600 \
        "./primus-cli direct -- benchmark gemm-deepseek --duration 5 --output-file ${OUT_IN_CTR}/gemm-deepseek_N${N}.md"

    run attention $N 1200 \
        "./primus-cli direct -- benchmark attention --backend flash --mbs-list 4 --report-csv-path ${OUT_IN_CTR}/attention_N${N}.csv"

    if [[ $N -ge 2 ]]; then
        run rccl $N 900 \
            "./primus-cli direct -- benchmark rccl --op all_reduce --output-file ${OUT_IN_CTR}/rccl_N${N}.md"
    else
        echo "----- rccl N=1 SKIPPED (collective needs N>=2) -----" | tee -a "$SUMMARY"
    fi

    GBS=$((4 * N))
    run megatron-llama2_7B-bf16 $N 1800 \
        "./primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml global_batch_size=${GBS}"
done

echo                                          | tee -a "$SUMMARY"
echo "Finished   : $(date -Iseconds)"        | tee -a "$SUMMARY"
