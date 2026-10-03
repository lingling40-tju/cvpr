#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stop_pair_group4_20261004"
run="$root/runlogs/stop_pair_service"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'STOP-pair Habitat startup already active' >&2; exit 2; }
test "$(sha256sum "$root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  d2420040cede386967f2203df48af16cf1dd2cac7a9732b662071eab0021eabd
if curl -fsS --max-time 2 http://127.0.0.1:5036/health >/dev/null 2>&1; then
  test -s "$run/habitat.pid"
  kill -0 "$(cat "$run/habitat.pid")"
  exit 0
fi
cd "$root"
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_stop_pair_5036 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port=5036 'vlnce.gpus=[0]' \
  'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/habitat.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/habitat.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5036/health >"$run/health.json" 2>/dev/null; then
    echo 'isolated STOP-pair Habitat service healthy'
    exit 0
  fi
  if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2
    exit 1
  fi
  sleep 2
done
echo 'STOP-pair Habitat service did not become healthy' >&2
exit 1
