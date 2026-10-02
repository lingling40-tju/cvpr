#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/fused_services"
mkdir -p "$run"

if ! curl -fsS --max-time 2 http://127.0.0.1:8021/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$base/activevln_train_env/bin/python" -u tools/reward_server.py \
    --port 8021 \
    --nav-model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
    --siglip-model "$base/models/siglip-base-patch16-224" \
    --temporal-checkpoint "$source_root/runlogs/ordinal_progress/policy_preference/temporal_contrastive_v2/encoder.pt" \
    --siglip-adapter "$source_root/runlogs/ordinal_progress/siglip_lora_goal_policy_fixed64_768_seed11/adapter.pt" \
    --development-report "$source_root/runlogs/ordinal_progress/policy_preference/equal_fused_reward_online_parity_development.json" \
    >"$run/reward.log" 2>&1 </dev/null &
  echo $! >"$run/reward.pid"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8021/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$(cat "$run/reward.pid")" 2>/dev/null; then
      tail -40 "$run/reward.log" >&2
      exit 1
    fi
    sleep 2
  done
fi
curl -fsS --max-time 3 http://127.0.0.1:8021/health >"$run/reward_health.json"

if ! curl -fsS --max-time 2 http://127.0.0.1:5021/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/tmp/td_srv_fused_5021 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5021 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null &
  echo $! >"$run/habitat.pid"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:5021/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
      tail -40 "$run/habitat.log" >&2
      exit 1
    fi
    sleep 2
  done
fi
curl -fsS --max-time 3 http://127.0.0.1:5021/health >"$run/habitat_health.json"
echo 'fused reward and Habitat services healthy'
