#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stop_boundary_20261005"
prior="$base/ActiveVLN_turnwise_oracle_20261004"
run="$root/runlogs/stop_boundary_service"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'boundary Habitat startup already active' >&2; exit 2; }
test -f "$prior/runlogs/oracle_exact512_scale/suite.completed"
test ! -f "$prior/runlogs/oracle_exact512_scale/suite.failed"
test "$(sha256sum "$root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  63a7ec72c5fef68fed7e8a7ea384a829fecf5d9665673151b1c6314009182661
test "$(sha256sum "$root/vlnce_server/semantic_reward/stop_boundary_reward.py" | awk '{print $1}')" = \
  8f3e37b89e4e903d0e10a7ec14ae64f784bb76835b93e310865b2cd75572b0b1
if curl -fsS --max-time 2 http://127.0.0.1:5036/health >/dev/null 2>&1; then
  test -s "$run/habitat.pid"
  pid=$(cat "$run/habitat.pid")
  ps -o args= -p "$pid" | grep -F 'server.port=5036' >/dev/null
  exit 0
fi
used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
test -n "$used0" && test "$used0" -lt 8000
cd "$root"
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  VLN_ORACLE_TURNWISE=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_stop_boundary_5036 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port=5036 'vlnce.gpus=[0]' \
  'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/habitat.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/habitat.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5036/health >"$run/health.json" 2>/dev/null; then
    echo 'isolated stop-boundary Habitat service healthy'
    exit 0
  fi
  if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2
    exit 1
  fi
  sleep 2
done
echo 'stop-boundary Habitat service did not become healthy' >&2
exit 1
