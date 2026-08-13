#!/bin/bash
# Once the rocm/primus:v25.9_gfx950 pull finishes:
#   1. Convert to SIF
#   2. Rerun Megatron llama2_7B BF16 sweep N=1..8 with primus_turbo enabled
#      (image-side primus_turbo built for gfx950 / MI355X)
#   3. Regenerate REPORT.md
set -uo pipefail

PRIMUS_DIR=/home/v89592/shaohao/primus/Primus
NEW_SIF=/home/v89592/shaohao/primus/image/primus-v25.9_gfx950.sif
RUN_ID=20260615-222308
BASE=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
OUT_IN_CTR=/workspace/sweep_out_${RUN_ID}
SUMMARY="${BASE}/summary.txt"
REPORT=/home/v89592/shaohao/primus/REPORT.md
B200=/home/v89592/shaohao/megatron-lm/work/summary.md
PULL_LOG=/home/v89592/shaohao/primus/logs/gfx950-pull.out

echo                                                       | tee -a "$SUMMARY"
echo "================ MEGATRON gfx950 RERUN $(date -Iseconds) ================" \
                                                            | tee -a "$SUMMARY"

# -- 1. Wait for image pull --
echo "[ORCH] $(date -Iseconds) waiting for pull of rocm/primus:v25.9_gfx950" | tee -a "$SUMMARY"
WAIT_START=$(date +%s)
while ! grep -q '^PULL_RC=' "$PULL_LOG" 2>/dev/null; do
    sleep 30
    if (( $(date +%s) - WAIT_START > 3600 )); then
        echo "[ORCH] pull timeout (>1h)" | tee -a "$SUMMARY"
        exit 1
    fi
done
PULL_RC=$(grep '^PULL_RC=' "$PULL_LOG" | tail -1 | cut -d= -f2)
echo "[ORCH] $(date -Iseconds) pull done PULL_RC=${PULL_RC}" | tee -a "$SUMMARY"
if [[ "$PULL_RC" != "0" ]]; then
    echo "[ORCH] pull failed; aborting" | tee -a "$SUMMARY"
    tail -50 "$PULL_LOG" | tee -a "$SUMMARY"
    exit 1
fi

# -- 2. Convert to SIF (skip if exists) --
if [[ ! -f "$NEW_SIF" ]]; then
    echo "[ORCH] $(date -Iseconds) converting to SIF" | tee -a "$SUMMARY"
    cd /home/v89592/shaohao/primus/image
    TMPDIR=/home/v89592/shaohao/primus/image \
    APPTAINER_TMPDIR=/home/v89592/shaohao/primus/image \
    APPTAINER_CACHEDIR=/home/v89592/shaohao/primus/image/.apptainer-cache \
    bash -c "set -e; \
        podman save -o primus-v25.9_gfx950.tar docker.io/rocm/primus:v25.9_gfx950; \
        singularity build primus-v25.9_gfx950.sif docker-archive://primus-v25.9_gfx950.tar; \
        rm -f primus-v25.9_gfx950.tar; rm -rf .apptainer-cache" \
        2>&1 | tee -a "$SUMMARY"
fi

if [[ ! -f "$NEW_SIF" ]]; then
    echo "[ORCH] SIF missing after convert; aborting" | tee -a "$SUMMARY"
    exit 1
fi
echo "[ORCH] $(date -Iseconds) using SIF: $(ls -l "$NEW_SIF" | awk '{print $5,$9}')" \
    | tee -a "$SUMMARY"

# -- 3. Run Megatron sweep --
run_megatron() {
    local N=$1
    local devs port log start rc dur status GBS
    devs=$(seq -s, 0 $((N-1)))
    port=$((29500 + (RANDOM % 500) + N))
    log="${BASE}/megatron-llama2_7B-bf16_N${N}.log"
    GBS=$((4 * N))
    start=$(date +%s)

    echo "----- GFX950 RERUN megatron-llama2_7B-bf16 N=${N} port=${port} devs=${devs} $(date -Iseconds) -----" \
        | tee -a "$SUMMARY"

    timeout --signal=TERM --kill-after=30s 1800 \
        singularity exec --rocm --writable-tmpfs \
            --pwd /workspace \
            --bind "${PRIMUS_DIR}":/workspace \
            --env HIP_VISIBLE_DEVICES="${devs}" \
            --env ROCR_VISIBLE_DEVICES="${devs}" \
            --env GPUS_PER_NODE="${N}" \
            --env NNODES=1 --env NODE_RANK=0 \
            --env MASTER_ADDR=localhost --env MASTER_PORT="${port}" \
            "${NEW_SIF}" \
            bash -c "./primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml global_batch_size=${GBS}" \
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

# -- 4. Regenerate report --
echo "[ORCH] $(date -Iseconds) regenerating $REPORT" | tee -a "$SUMMARY"
/usr/bin/python3.11 /home/v89592/shaohao/primus/generate_report.py \
    "${BASE}" \
    "/home/v89592/shaohao/primus/Primus/sweep_out_${RUN_ID}" \
    "${B200}" \
    "${REPORT}" 2>&1 | tee -a "$SUMMARY"
echo "[ORCH] $(date -Iseconds) DONE" | tee -a "$SUMMARY"
