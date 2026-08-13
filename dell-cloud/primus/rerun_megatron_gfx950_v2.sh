#!/bin/bash
# Megatron sweep using rocm/primus:v25.9_gfx950 SIF.
# Uses the image's own bundled Primus (no bind from host).
# Patches model YAML in writable-tmpfs at runtime to swap Llama2 tokenizer ->
# NullTokenizer (no HF download needed) so mock_data path works offline.
set -uo pipefail

NEW_SIF=/home/v89592/shaohao/primus/image/primus-v25.9_gfx950.sif
RUN_ID=20260615-222308
BASE=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
SUMMARY="${BASE}/summary.txt"
REPORT=/home/v89592/shaohao/primus/REPORT.md
B200=/home/v89592/shaohao/megatron-lm/work/summary.md
BENCH_OUT=/home/v89592/shaohao/primus/Primus/sweep_out_${RUN_ID}

echo                                                              | tee -a "$SUMMARY"
echo "================ MEGATRON gfx950 v2 RERUN $(date -Iseconds) ================" \
                                                                   | tee -a "$SUMMARY"

run_megatron() {
    local N=$1
    local devs port log start rc dur status GBS
    devs=$(seq -s, 0 $((N-1)))
    port=$((29500 + (RANDOM % 500) + N))
    log="${BASE}/megatron-llama2_7B-bf16_N${N}.log"
    GBS=$((4 * N))
    start=$(date +%s)

    echo "----- GFX950v2 megatron N=${N} port=${port} devs=${devs} $(date -Iseconds) -----" \
        | tee -a "$SUMMARY"

    timeout --signal=TERM --kill-after=30s 1800 \
        singularity exec --rocm \
            --overlay /home/v89592/shaohao/primus/overlay-megatron.img \
            --pwd /workspace/Primus \
            --env EXP=examples/megatron/configs/llama2_7B-pretrain.yaml \
            --env HIP_VISIBLE_DEVICES="${devs}" \
            --env ROCR_VISIBLE_DEVICES="${devs}" \
            --env GPUS_PER_NODE="${N}" \
            --env NNODES=1 --env NODE_RANK=0 \
            --env MASTER_ADDR=localhost --env MASTER_PORT="${port}" \
            --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
            "${NEW_SIF}" \
            bash -c '
                MY=/workspace/Primus/primus/configs/models/megatron/llama2_7B.yaml
                sed -i "s|tokenizer_type: Llama2Tokenizer|tokenizer_type: NullTokenizer|" "$MY"
                sed -i "/^tokenizer_model:/d" "$MY"
                grep -q "^vocab_size:" "$MY" || echo "vocab_size: 32000" >> "$MY"
                bash examples/run_pretrain.sh global_batch_size='"$GBS"'
            ' \
        > "${log}" 2>&1
    rc=$?
    dur=$(($(date +%s) - start))
    if   [[ $rc -eq 0   ]]; then status="OK"
    elif [[ $rc -eq 124 ]]; then status="TIMEOUT(1800s)"
    else                         status="FAIL(rc=${rc})"
    fi
    echo "  ${status}  duration=${dur}s  log=${log}" | tee -a "$SUMMARY"
}

for N in 1 2 3 4 5 6 7 8; do
    run_megatron $N
done

echo "[ORCH-v2] $(date -Iseconds) regenerating $REPORT" | tee -a "$SUMMARY"
/usr/bin/python3.11 /home/v89592/shaohao/primus/generate_report.py \
    "${BASE}" "${BENCH_OUT}" "${B200}" "${REPORT}" 2>&1 | tee -a "$SUMMARY"
echo "[ORCH-v2] $(date -Iseconds) DONE" | tee -a "$SUMMARY"
