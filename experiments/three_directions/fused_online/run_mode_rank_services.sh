#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_mode_rank_20261003"
source_root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/mode_rank_services"
mkdir -p "$run"
exec 9>"$run/services.lock"
flock -n 9 || { echo 'mode-rank service startup already running' >&2; exit 2; }

start_reward() {
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1; then
    test -s "$run/reward.pid"
    ps -o args= -p "$(cat "$run/reward.pid")" | grep -F 'failure_only_reward_server.py' >/dev/null
    return
  fi
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$source_root/tools:$source_root:$root${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_train_env/bin/python" -u \
    tools/failure_only_reward_server.py --port 8026 \
    --nav-model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
    --expected-nav-config-sha256 \
      9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9 \
    --temporal-checkpoint "$source_root/runlogs/failure_rank/encoder.pt" \
    --calibration "$source_root/runlogs/failure_rank/reward_calibration.json" \
    >"$run/reward.log" 2>&1 </dev/null &
  echo $! >"$run/reward.pid"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$(cat "$run/reward.pid")" 2>/dev/null; then
      tail -40 "$run/reward.log" >&2; exit 1
    fi
    sleep 2
  done
  curl -fsS --max-time 3 http://127.0.0.1:8026/health >"$run/reward_health.json"
  "$base/activevln_train_env/bin/python" - "$run/reward_health.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='ok' and x['reward_variant']=='failure_only_temporal_v1'
assert x['encoder_sha256']=='b09348dd75520e7a00cfa036db197b6759ecd67ba64b4c92898bf2532f0bb0d2'
PY
}

start_habitat() {
  if curl -fsS --max-time 2 http://127.0.0.1:5026/health >/dev/null 2>&1; then
    test -s "$run/habitat.pid"
    ps -o args= -p "$(cat "$run/habitat.pid")" | grep -F 'server.port=5026' >/dev/null
    return
  fi
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/tmp/td_srv_mode_rank_5026 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5026 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null &
  echo $! >"$run/habitat.pid"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:5026/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
      tail -40 "$run/habitat.log" >&2; exit 1
    fi
    sleep 2
  done
  curl -fsS --max-time 3 http://127.0.0.1:5026/health >"$run/habitat_health.json"
}

start_reward
start_habitat
echo 'mode-rank reward and Habitat services healthy'
