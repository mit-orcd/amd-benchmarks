#!/usr/bin/env bash
# Re-run ONLY the bandwidth-sensitive RVS health modules, on a quiet machine.
#
# Why this exists: the first health pass (health_20260813_195057) overlapped a ~1.5 TB
# model download writing to /mnt/scratch at ~1 GB/s. NVMe writes and NIC traffic both
# consume PCIe lanes and host memory bandwidth, so `pebb` (host<->device PCIe) and to a
# lesser degree `babel` (HBM) and `pbqt` (P2P XGMI) may read low in that pass.
#
# These three modules are minutes each, so re-running them clean is cheap insurance --
# and it matters, because Part B's cliff attribution ("algorithm, not fabric") rests on
# pbqt coming back healthy.
#
# Usage: ./rerun_bandwidth_health.sh
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
RVS=$RVS_BIN
CONF=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/share/rocm-validation-suite/conf
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rvs/health_bw_$TS; mkdir -p "$OUT"
SUM=$OUT/health_bw_summary.txt

# ---- refuse to run dirty -------------------------------------------------------
busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
if [[ "${busy:-0}" -ne 0 ]]; then
  echo "ABORT: $busy GPU(s) busy." >&2; exit 1
fi
if pgrep -f 'hf download|download_models.sh' >/dev/null 2>&1; then
  echo "ABORT: a model download is still running — that is the exact contention this" >&2
  echo "       script exists to avoid. Wait for it to finish." >&2
  exit 1
fi
# Any sustained writer to /mnt/scratch would do the same damage.
if [[ $(awk '/nvme/ {print $10}' /proc/diskstats | paste -sd+ | bc 2>/dev/null || echo 0) -lt 0 ]]; then :; fi

MODULES=("babel:babel.conf:HBM bandwidth"
         "pebb:pebb_single.conf:PCIe host<->device bandwidth"
         "pbqt:pbqt_single.conf:peer-to-peer XGMI bandwidth")

resolve_conf() {
  local mod=$1 base=$2
  for c in "$CONF/MI355X/$base" "$CONF/$base"; do
    [[ -f "$c" ]] && { echo "$c"; return 0; }
  done
  find "$CONF" -maxdepth 2 -name "${mod}_*.conf" 2>/dev/null | head -1
}

{ echo "RVS bandwidth-module re-run $TS (quiet machine)"; echo "rvs: $RVS"; echo; } | tee "$SUM"
for entry in "${MODULES[@]}"; do
  IFS=: read -r mod base desc <<<"$entry"
  c=$(resolve_conf "$mod" "$base"); log="$OUT/${mod}.log"
  [[ -n "$c" && -f "$c" ]] || { echo "SKIP $mod (no conf)" | tee -a "$SUM"; continue; }
  echo "----- $mod ($desc) conf=${c#$CONF/} -----" | tee -a "$SUM"
  timeout 900 "$RVS" -c "$c" -d 3 >"$log" 2>&1
  echo "  rc=$? log=$log" | tee -a "$SUM"
done
echo "results: $OUT" | tee -a "$SUM"
echo
echo "Compare against the first pass to see whether download contention mattered:"
echo "  diff <(grep -oE '[0-9.]+ GB/s' $LOG_ROOT/rvs/health_*/pebb.log | sort) \\"
echo "       <(grep -oE '[0-9.]+ GB/s' $OUT/pebb.log | sort)"
