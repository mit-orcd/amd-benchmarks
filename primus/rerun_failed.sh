#!/bin/bash
# Rerun the failed benches (megatron + rccl) from sweep 20260615-222308,
# then regenerate REPORT.md. Outputs land in the same sweep dirs (overwriting
# the failed logs/files), so generate_report.py picks them up automatically.
set -uo pipefail

PRIMUS_DIR=/home/v89592/shaohao/primus/Primus
SIF=/home/v89592/shaohao/primus/image/primus-v26.3.sif
RUN_ID=20260615-222308
BASE=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
OUT_IN_CTR=/workspace/sweep_out_${RUN_ID}
SUMMARY="${BASE}/summary.txt"
REPORT=/home/v89592/shaohao/primus/REPORT.md
B200=/home/v89592/shaohao/megatron-lm/work/summary.md

echo                                                | tee -a "$SUMMARY"
echo "================ RERUN $(date -Iseconds) ================" | tee -a "$SUMMARY"

run() {
    local name=$1 N=$2 timeout_s=$3 cmd=$4
    local devs port log start rc dur status
    devs=$(seq -s, 0 $((N-1)))
    port=$((29500 + (RANDOM % 500) + N))
    log="${BASE}/${name}_N${N}.log"
    start=$(date +%s)

    echo "----- RERUN ${name} N=${N} port=${port} devs=${devs} $(date -Iseconds) -----" \
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

# ----- Megatron first (per user request) -----
TURBO_OFF="enable_primus_turbo=false use_turbo_attention=false use_turbo_grouped_mlp=false"

for N in 1 2 3 4 5 6 7 8; do
    GBS=$((4 * N))
    run megatron-llama2_7B-bf16 $N 1800 \
        "./primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml global_batch_size=${GBS} ${TURBO_OFF}"
done

# ----- RCCL (default op = allreduce) -----
for N in 2 3 4 5 6 7 8; do
    run rccl $N 900 \
        "./primus-cli direct -- benchmark rccl --output-file ${OUT_IN_CTR}/rccl_N${N}.md"
done

echo "Rerun finished : $(date -Iseconds)" | tee -a "$SUMMARY"

# ----- Regenerate report -----
echo "[rerun] $(date -Iseconds) regenerating $REPORT" | tee -a "$SUMMARY"
/usr/bin/python3.11 /home/v89592/shaohao/primus/generate_report.py \
    "${BASE}" \
    "/home/v89592/shaohao/primus/Primus/sweep_out_${RUN_ID}" \
    "${B200}" \
    "${REPORT}" 2>&1 | tee -a "$SUMMARY"
echo "[rerun] $(date -Iseconds) DONE" | tee -a "$SUMMARY"
