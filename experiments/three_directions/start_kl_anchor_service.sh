#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
port=5014
sim_gpu=${1:-1}
[[ "$sim_gpu" =~ ^[0-3]$ ]]
run_dir="$root/runlogs/kl_anchor_service"
mkdir -p "$run_dir"
pid_file="$run_dir/server.pid"
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  test -s "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null
  echo "existing KL service healthy on $port"
  exit 0
fi
cd "$root"
nohup env PYTHONUNBUFFERED=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/tmp/td_srv_kl_anchor_5014 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  "server.port=$port" "vlnce.gpus=[$sim_gpu]" \
  'vlnce.r2r_gpu_plan=[8]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run_dir/server.log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "KL service healthy on $port (PID $pid, GPU $sim_gpu, 8 R2R simulators)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
echo 'KL service failed; inspect server.log' >&2
exit 1
