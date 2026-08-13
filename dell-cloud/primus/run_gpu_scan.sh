#!/bin/bash
# Sweep Primus benchmark gemm across 1..8 GPUs inside the singularity image.
set -uo pipefail

PRIMUS_DIR=/home/v89592/shaohao/primus/Primus
SIF=/home/v89592/shaohao/primus/image/primus-v26.3.sif
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/home/v89592/shaohao/primus/logs/gpu-scan-${RUN_ID}
SUMMARY=${LOG_DIR}/summary.txt
PER_RUN_TIMEOUT=900  # 15 min per GPU count

mkdir -p "$LOG_DIR"
echo "Primus GPU-scan run ${RUN_ID}"  | tee "$SUMMARY"
echo "SIF       : $SIF"               | tee -a "$SUMMARY"
echo "Repo      : $PRIMUS_DIR"        | tee -a "$SUMMARY"
echo "Log dir   : $LOG_DIR"           | tee -a "$SUMMARY"
echo "Started   : $(date -Iseconds)"  | tee -a "$SUMMARY"
echo                                   | tee -a "$SUMMARY"

for N in 1 2 3 4 5 6 7 8; do
    DEVS=$(seq -s, 0 $((N-1)))
    LOG="${LOG_DIR}/scan_${N}gpu.log"
    PORT=$((29500 + N))
    START=$(date +%s)

    echo "===== ${N} GPU(s) — visible=${DEVS} — port=${PORT} — $(date -Iseconds) =====" \
        | tee -a "$SUMMARY"

    timeout --signal=TERM --kill-after=30s "${PER_RUN_TIMEOUT}" \
        singularity exec --rocm \
            --pwd /workspace \
            --bind "${PRIMUS_DIR}":/workspace \
            --env HIP_VISIBLE_DEVICES="${DEVS}" \
            --env ROCR_VISIBLE_DEVICES="${DEVS}" \
            --env GPUS_PER_NODE="${N}" \
            --env NNODES=1 \
            --env NODE_RANK=0 \
            --env MASTER_ADDR=localhost \
            --env MASTER_PORT="${PORT}" \
            "${SIF}" \
            bash -c './primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --duration 10' \
        >"$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    DUR=$((END - START))

    if [[ $RC -eq 0 ]]; then
        STATUS="OK"
    elif [[ $RC -eq 124 ]]; then
        STATUS="TIMEOUT(${PER_RUN_TIMEOUT}s)"
    else
        STATUS="FAIL(rc=$RC)"
    fi
    echo "  -> ${STATUS}  duration=${DUR}s  log=${LOG}" | tee -a "$SUMMARY"
done

echo                                  | tee -a "$SUMMARY"
echo "Finished : $(date -Iseconds)"   | tee -a "$SUMMARY"
