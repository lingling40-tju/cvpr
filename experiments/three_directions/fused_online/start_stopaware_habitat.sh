#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
run="$root/runlogs/stopaware_services"
mkdir -p "$run"
test "$(sha256sum "$root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  e3644b522cf630f489da049eb6766eb02835cb6fda898a494e951d60e4048051
if curl -fsS --max-time 2 http://127.0.0.1:5021/health >/dev/null 2>&1; then
  test -s "$run/habitat.pid"
  pid=$(cat "$run/habitat.pid")
  kill -0 "$pid"
  ps -o args= -p "$pid" | grep -F 'server.port=5021' >/dev/null
else
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/tmp/td_srv_stopaware_5021 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5021 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null &
  echo $! >"$run/habitat.pid"
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:5021/health >/dev/null 2>&1; then
      ready=1; break
    fi
    if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
      tail -40 "$run/habitat.log" >&2
      exit 1
    fi
    sleep 2
  done
  test "$ready" -eq 1
fi
curl -fsS --max-time 3 http://127.0.0.1:5021/health >"$run/habitat_health.json"
curl -fsS --max-time 3 http://127.0.0.1:8024/health >"$run/reward_health.json"
"$base/activevln_train_env/bin/python" - "$run/reward_health.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='ok' and x['reward_variant']=='failure_only_temporal_v1'
assert x['encoder_sha256']=='b09348dd75520e7a00cfa036db197b6759ecd67ba64b4c92898bf2532f0bb0d2'
PY
echo 'stop-aware Habitat and frozen potential scorer healthy'
