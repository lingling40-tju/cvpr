#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
run="$root/runlogs/oracle_turnwise_service"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'oracle Habitat startup already active' >&2; exit 2; }
test "$(sha256sum "$root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  f0240db71d20fc8ab1afa9eb0c9ccb61312b5e9a5aa451430747e55266afcf43
if curl -fsS --max-time 2 http://127.0.0.1:5035/health >/dev/null 2>&1; then
  test -s "$run/habitat.pid"
  kill -0 "$(cat "$run/habitat.pid")"
  exit 0
fi
cd "$root"
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  VLN_ORACLE_TURNWISE=1 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_oracle_turnwise_5035 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port=5035 'vlnce.gpus=[0]' \
  'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/habitat.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/habitat.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5035/health >"$run/health.json" 2>/dev/null; then
    echo 'isolated oracle-turnwise Habitat service healthy'
    exit 0
  fi
  if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2
    exit 1
  fi
  sleep 2
done
echo 'oracle Habitat service did not become healthy' >&2
exit 1
