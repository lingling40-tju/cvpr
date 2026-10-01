#!/usr/bin/env bash
set -euo pipefail

# Starts only the isolated progress-reward simulator. Do not reuse an active
# branch training service, whose workers imported the original reward module.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_progress_fallback_20261002"
source_root="$base/ActiveVLN_three_directions_20261002"
sim_gpu=${1:-3}
port=5011
[[ "$sim_gpu" =~ ^[0-3]$ ]] || { echo 'GPU index must be 0..3' >&2; exit 2; }
test -f "$source_root/runlogs/three_direction_scale_branch_128step_full_eval_all/suite.completed"
test -s "$root/progress_source.txt"
mkdir -p "$root/runlogs"
pid_file="$root/runlogs/progress_service.pid"
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  if test -f "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "existing progress service healthy on $port"
    exit 0
  fi
  echo "port $port is already occupied by an untracked service" >&2
  exit 2
fi

cd "$root"
nohup env PYTHONUNBUFFERED=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/tmp/td_srv_progress_5011 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  "server.port=$port" "vlnce.gpus=[$sim_gpu]" \
  'vlnce.r2r_gpu_plan=[8]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$root/runlogs/progress_service.log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
for attempt in $(seq 1 90); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "progress service healthy on $port (PID $pid, GPU $sim_gpu)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
echo "progress service failed; inspect $root/runlogs/progress_service.log" >&2
exit 1
