#!/bin/bash
# Rerun Megatron for N=3,5,6,7 (failed because default GBS=256 doesn't divide
# by MBS=4 * DP=N). Patches the OVERRIDE in llama2_7B-pretrain.yaml to a
# per-N GBS close to 256:
#   N=3 -> GBS=252 (21 micro-batches), N=5 -> 240, N=6 -> 240, N=7 -> 252
# Restores GBS=256 at the end so the overlay is clean for future runs.
set -uo pipefail

NEW_SIF=/home/v89592/shaohao/primus/image/primus-v25.9_gfx950.sif
OVERLAY=/home/v89592/shaohao/primus/overlay-megatron.img
RUN_ID=20260615-222308
BASE=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
SUMMARY="${BASE}/summary.txt"
REPORT=/home/v89592/shaohao/primus/REPORT.md
B200=/home/v89592/shaohao/megatron-lm/work/summary.md
BENCH_OUT=/home/v89592/shaohao/primus/Primus/sweep_out_${RUN_ID}
PRETRAIN_YAML=/workspace/Primus/examples/megatron/configs/llama2_7B-pretrain.yaml

echo                                                      | tee -a "$SUMMARY"
echo "================ MEGATRON missing-N rerun $(date -Iseconds) ================" \
                                                           | tee -a "$SUMMARY"

run_megatron() {
    local N=$1 GBS=$2
    local devs port log start rc dur status
    devs=$(seq -s, 0 $((N-1)))
    port=$((29500 + (RANDOM % 500) + N))
    log="${BASE}/megatron-llama2_7B-bf16_N${N}.log"
    start=$(date +%s)

    echo "----- MISSING-N megatron N=${N} GBS=${GBS} port=${port} devs=${devs} $(date -Iseconds) -----" \
        | tee -a "$SUMMARY"

    timeout --signal=TERM --kill-after=30s 1800 \
        singularity exec --rocm \
            --overlay "${OVERLAY}" \
            --pwd /workspace/Primus \
            --env EXP=examples/megatron/configs/llama2_7B-pretrain.yaml \
            --env HIP_VISIBLE_DEVICES="${devs}" \
            --env ROCR_VISIBLE_DEVICES="${devs}" \
            --env GPUS_PER_NODE="${N}" \
            --env NNODES=1 --env NODE_RANK=0 \
            --env MASTER_ADDR=localhost --env MASTER_PORT="${port}" \
            --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
            "${NEW_SIF}" \
            bash -c "
                MY=/workspace/Primus/primus/configs/models/megatron/llama2_7B.yaml
                # idempotent NullTokenizer patch (in case fresh overlay)
                sed -i 's|tokenizer_type: Llama2Tokenizer|tokenizer_type: NullTokenizer|' \"\$MY\"
                sed -i '/^tokenizer_model:/d' \"\$MY\"
                grep -q '^vocab_size:' \"\$MY\" || echo 'vocab_size: 32000' >> \"\$MY\"
                # patch global_batch_size override (line is indented under 'overrides:')
                sed -i 's|^\\([[:space:]]*\\)global_batch_size:.*|\\1global_batch_size: ${GBS}|' \
                    ${PRETRAIN_YAML}
                bash examples/run_pretrain.sh
            " \
        > "${log}" 2>&1
    rc=$?
    dur=$(($(date +%s) - start))
    if   [[ $rc -eq 0   ]]; then status="OK"
    elif [[ $rc -eq 124 ]]; then status="TIMEOUT(1800s)"
    else                         status="FAIL(rc=${rc})"
    fi
    echo "  ${status}  duration=${dur}s  log=${log}" | tee -a "$SUMMARY"
}

run_megatron 3 252
run_megatron 5 240
run_megatron 6 240
run_megatron 7 252

# Restore GBS=256 so the overlay matches the original config for future runs.
echo "[missing-N] restoring pretrain YAML global_batch_size to 256" | tee -a "$SUMMARY"
singularity exec --rocm --overlay "${OVERLAY}" "${NEW_SIF}" \
    bash -c "sed -i 's|^\\([[:space:]]*\\)global_batch_size:.*|\\1global_batch_size: 256|' \
             ${PRETRAIN_YAML}" 2>&1 | tee -a "$SUMMARY"

echo "[missing-N] $(date -Iseconds) regenerating $REPORT" | tee -a "$SUMMARY"
/usr/bin/python3.11 /home/v89592/shaohao/primus/generate_report.py \
    "${BASE}" "${BENCH_OUT}" "${B200}" "${REPORT}" 2>&1 | tee -a "$SUMMARY"
echo "[missing-N] $(date -Iseconds) DONE" | tee -a "$SUMMARY"
