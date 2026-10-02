#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_route_fidelity_20261002"
source_root="$base/ActiveVLN_three_directions_20261002"
sim_gpu=${1:-2}
port=5012
[[ "$sim_gpu" =~ ^[0-3]$ ]] || { echo 'GPU index must be 0..3' >&2; exit 2; }
test -f "$source_root/runlogs/three_direction_progress_scale_conditional/suite.completed"
test -s "$root/route_source.txt"
mkdir -p "$root/runlogs"
pid_file="$root/runlogs/route_service.pid"
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  if test -f "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "existing route service healthy on $port"
    exit 0
  fi
  echo "port $port is already occupied by an untracked service" >&2
  exit 2
fi

cd "$root"
nohup env PYTHONUNBUFFERED=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/tmp/td_srv_route_5012 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  "server.port=$port" "vlnce.gpus=[$sim_gpu]" \
  'vlnce.r2r_gpu_plan=[8]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$root/runlogs/route_service.log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
for attempt in $(seq 1 90); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "route service healthy on $port (PID $pid, GPU $sim_gpu)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
echo "route service failed; inspect $root/runlogs/route_service.log" >&2
exit 1
