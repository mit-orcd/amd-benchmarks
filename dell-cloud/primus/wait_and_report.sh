#!/bin/bash
# Wait for the active Primus sweep to finish, then auto-generate REPORT.md.
# Intended to run alongside run_full_sweep.sh in nohup.
set -uo pipefail

RUN_ID="$1"
SWEEP_DIR=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}
BENCH_OUT=/home/v89592/shaohao/primus/Primus/sweep_out_${RUN_ID}
B200=/home/v89592/shaohao/megatron-lm/work/summary.md
REPORT=/home/v89592/shaohao/primus/REPORT.md
WATCH_LOG=/home/v89592/shaohao/primus/logs/sweep-${RUN_ID}/watcher.log

mkdir -p "$SWEEP_DIR"
exec >"$WATCH_LOG" 2>&1

echo "[watcher] $(date -Iseconds) waiting for $SWEEP_DIR/summary.txt to contain 'Finished'"
START=$(date +%s)
MAX_WAIT=$((6 * 3600))   # 6 hour hard cap

while true; do
    if [[ -f "$SWEEP_DIR/summary.txt" ]] && grep -q '^Finished' "$SWEEP_DIR/summary.txt"; then
        echo "[watcher] $(date -Iseconds) sweep finished"
        break
    fi
    NOW=$(date +%s)
    if (( NOW - START > MAX_WAIT )); then
        echo "[watcher] $(date -Iseconds) max wait exceeded — generating report with whatever data exists"
        break
    fi
    # Also bail if no sweep process is alive AND summary.txt hasn't grown for >5 min
    if ! pgrep -af 'run_full_sweep|singularity exec.*primus-v26.3' >/dev/null 2>&1; then
        echo "[watcher] $(date -Iseconds) no sweep process detected; will check once more in 60s"
        sleep 60
        if ! pgrep -af 'run_full_sweep|singularity exec.*primus-v26.3' >/dev/null 2>&1; then
            echo "[watcher] $(date -Iseconds) sweep no longer running — generating report"
            break
        fi
    fi
    sleep 30
done

echo "[watcher] $(date -Iseconds) generating report -> $REPORT"
python3 /home/v89592/shaohao/primus/generate_report.py \
    "$SWEEP_DIR" "$BENCH_OUT" "$B200" "$REPORT" \
    && echo "[watcher] $(date -Iseconds) REPORT WRITTEN" \
    || echo "[watcher] $(date -Iseconds) report generation FAILED rc=$?"
