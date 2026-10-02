#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/failure_only_service"
mkdir -p "$run"
test -f "$root/runlogs/fused_scale_conditional/no_pilot_gain"
test -f "$root/runlogs/fused_scale_conditional/suite.completed"
if ! curl -fsS --max-time 2 http://127.0.0.1:8024/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$source_root/tools:$source_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_train_env/bin/python" -u \
    tools/failure_only_reward_server.py \
    --port 8024 \
    --nav-model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
    --expected-nav-config-sha256 \
      9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9 \
    --temporal-checkpoint "$root/runlogs/failure_rank/encoder.pt" \
    --calibration "$root/runlogs/failure_rank/reward_calibration.json" \
    >"$run/service.log" 2>&1 </dev/null &
  echo $! >"$run/service.pid"
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8024/health >/dev/null 2>&1; then
      ready=1; break
    fi
    if ! kill -0 "$(cat "$run/service.pid")" 2>/dev/null; then
      tail -40 "$run/service.log" >&2
      exit 1
    fi
    sleep 2
  done
  test "$ready" -eq 1
fi
curl -fsS --max-time 3 http://127.0.0.1:8024/health >"$run/health.json"
"$base/activevln_train_env/bin/python" - "$run/health.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='ok' and x['reward_variant']=='failure_only_temporal_v1'
assert x['encoder_sha256']=='b09348dd75520e7a00cfa036db197b6759ecd67ba64b4c92898bf2532f0bb0d2'
PY
echo 'failure-only reward service healthy'
