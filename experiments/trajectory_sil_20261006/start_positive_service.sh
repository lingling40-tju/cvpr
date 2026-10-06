#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
n8="$base/ActiveVLN_norm_terminal_rloo_n8_20261006/runlogs/posthoc_n8_pilot"
n4="$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/posthoc_n4_full1839"
run="$root/runlogs/service"
port=5085
test -f "$n8/suite.completed" && test ! -f "$n8/suite.failed"
test -f "$n4/suite.completed" && test ! -f "$n4/suite.failed"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'positive-trajectory service startup already active' >&2; exit 2; }
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  test -s "$run/server.pid"
  pid=$(cat "$run/server.pid")
  ps -o args= -p "$pid" | grep -F "server.port=$port" >/dev/null
  exit 0
fi
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "port $port is occupied" >&2
  exit 1
fi
mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$mem2" -lt 10000
cd "$root"
nohup env PYTHONUNBUFFERED=1 VLN_ORACLE_TURNWISE=0 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_positive_5085 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port="$port" 'vlnce.gpus=[2]' \
  'vlnce.r2r_gpu_plan=[32]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/server.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/server.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$run/health.json" 2>/dev/null; then
    echo "Habitat service healthy on $port"
    exit 0
  fi
  if ! kill -0 "$(cat "$run/server.pid")" 2>/dev/null; then
    tail -30 "$run/server.log" >&2
    exit 1
  fi
  sleep 2
done
echo 'Habitat service did not become healthy' >&2
exit 1
