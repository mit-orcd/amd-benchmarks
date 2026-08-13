#!/usr/bin/env bash
# Stop the ATOM server container started by run_atom_server.sh.
# Only ever touches OUR container by name -- never pkills by process pattern, because
# this box is shared and a name-based pkill would hit other users' servers.
set -uo pipefail
source /home/amd/shaohao/amd-benchmarks/amd-cloud/common/env.sh
NAME="${ATOM_CONTAINER:-atom-bench}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "No container named '$NAME'. Nothing to stop."
  exit 0
fi
echo "Stopping container '$NAME'..."
docker stop -t 30 "$NAME" >/dev/null 2>&1
docker rm "$NAME" >/dev/null 2>&1
echo "Stopped and removed '$NAME'."
rm -f "$LOG_ROOT/atom/CURRENT_PORT.txt"
sleep 3
rocm-smi --showmemuse 2>/dev/null | grep 'VRAM%' | head -8
