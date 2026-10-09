#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_direction_20261009"
run="$root/runlogs/service"
port=5086
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo "service startup locked" >&2; exit 2; }
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  test -s "$run/server.pid"
  ps -o args= -p "$(cat "$run/server.pid")" | grep -F "server.port=$port" >/dev/null
  exit 0
fi
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then echo "port $port occupied" >&2; exit 1; fi
mem=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$mem" -lt 10000
cd "$root"
nohup env PYTHONUNBUFFERED=1 VLN_ORACLE_TURNWISE=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_three_5086 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port="$port" 'vlnce.gpus=[2]' 'vlnce.r2r_gpu_plan=[32]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/server.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/server.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$run/health.json" 2>/dev/null; then
    echo "Habitat service healthy on $port"
    exit 0
  fi
  if ! kill -0 "$(cat "$run/server.pid")" 2>/dev/null; then tail -50 "$run/server.log" >&2; exit 1; fi
  sleep 2
done
echo "Habitat service timeout" >&2; exit 1
