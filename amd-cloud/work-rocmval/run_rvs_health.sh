#!/usr/bin/env bash
# RVS health modules: config, memory, HBM bandwidth, PCIe, P2P/XGMI, power.
#
# This is the "is the box healthy" gate that dell-cloud/rccl-tests/rccl-tests.md argues
# for: it validates the floor *beneath* RCCL, so a clean result here means a later
# RCCL cliff is an algorithm problem, not hardware.
#
# Deviations from plan.md as written, forced by what RVS 1.7.8 actually ships:
#   * There is no `pqt` module. Peer-to-peer / XGMI bandwidth is `pbqt` (P2P Benchmark
#     and Qualification Tool); `pebb` is host<->device. The plan's module list named a
#     `pqt` that does not exist and mislabelled pbqt as "PCIe bidirectional".
#   * Confs are resolved MI355X-first: conf/MI355X/<name> beats conf/<name> when present
#     (MI355X ships tuned babel/pebb/pbqt/gst/iet_stress confs).
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
RVS=$RVS_BIN
CONF=$BENCH_ROOT/work-rocmval/ROCmValidationSuite/install_local/share/rocm-validation-suite/conf
TS=$(date +%Y%m%d_%H%M%S); OUT=$LOG_ROOT/rvs/health_$TS; mkdir -p "$OUT"
SUM=$OUT/health_summary.txt

[[ -x "$RVS" ]] || { echo "ERROR: rvs not found at $RVS" >&2; exit 1; }

# module : conf basename : what it proves
MODULES=(
  "gpup:gpup_single.conf:GPU properties / config registers"
  "peqt:peqt_single.conf:PCIe qualification (capabilities match expectations)"
  "smqt:smqt_single.conf:SBIOS / VRAM BAR mapping"
  "rcqt:rcqt_single.conf:ROCm package + user/group checks"
  "mem:mem.conf:HBM error / pattern test"
  "babel:babel.conf:HBM bandwidth"
  "pebb:pebb_single.conf:PCIe host<->device bandwidth"
  "pbqt:pbqt_single.conf:peer-to-peer XGMI device<->device bandwidth"
  "iet:iet_single.conf:sustained power / EDP"
)

# MI355X-tuned conf wins over the generic one; fall back to any <mod>*.conf.
resolve_conf() {
  local mod=$1 base=$2
  for c in "$CONF/MI355X/$base" "$CONF/$base"; do
    [[ -f "$c" ]] && { echo "$c"; return 0; }
  done
  find "$CONF" -maxdepth 2 -name "${mod}_*.conf" -o -maxdepth 2 -name "${mod}.conf" 2>/dev/null | head -1
}

{ echo "RVS health run $TS"; echo "rvs : $RVS"; echo "conf: $CONF"; echo; } | tee "$SUM"

for entry in "${MODULES[@]}"; do
  IFS=: read -r mod base desc <<<"$entry"
  c=$(resolve_conf "$mod" "$base")
  log="$OUT/${mod}.log"
  [[ -n "$c" && -f "$c" ]] || { echo "SKIP $mod (no conf found)" | tee -a "$SUM"; continue; }
  echo "----- $mod ($desc) conf=${c#$CONF/} -----" | tee -a "$SUM"
  timeout 900 "$RVS" -c "$c" -d 3 >"$log" 2>&1
  rc=$?
  pass=$(grep -ci "pass.*true\|RESULT.*pass" "$log" 2>/dev/null | head -1)
  fail=$(grep -ci "RVS-ERROR\|FAIL" "$log" 2>/dev/null | head -1)
  echo "  rc=$rc pass_lines=${pass:-0} error_lines=${fail:-0} log=$log" | tee -a "$SUM"
done

# MI355X shipped level configs (levels 1..5, increasing coverage), if present.
for lvl in "$CONF"/MI355X/levels/rvs_level_*.conf; do
  [[ -f "$lvl" ]] || continue
  n=$(basename "$lvl" .conf)
  echo "----- $n -----" | tee -a "$SUM"
  timeout 1800 "$RVS" -c "$lvl" >"$OUT/$n.log" 2>&1
  echo "  rc=$? log=$OUT/$n.log" | tee -a "$SUM"
done

echo "results: $OUT" | tee -a "$SUM"
