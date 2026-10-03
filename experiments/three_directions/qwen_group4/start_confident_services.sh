#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_qwen_confident_20261004"
source_root="$base/ActiveVLN_three_directions_20261002"
qroot="$source_root/runlogs/ordinal_progress/qwen3_route_match"
run="$root/runlogs/confident_services"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'confidence service startup already running' >&2; exit 2; }
test -f "$base/ActiveVLN_qwen_group_rank_20261004/runlogs/qwen_group_scale/no_pilot_gain"

if ! curl -fsS --max-time 2 http://127.0.0.1:8033/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_semantic_verifier_env/bin/python" -u \
    tools/qwen_group_rank_server.py --port 8033 \
    --manifest "$qroot/online_exact256_manifest.json" \
    --pilot-parquet "$root/data/qwen3_group4_exact256.parquet" \
    --train-dataset "$base/ActiveVLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
    --model "$base/models/Qwen3-VL-8B-Instruct" \
    --model-hashes "$qroot/model_sha256.txt" \
    --expert-analysis "$qroot/expert_calibration_analysis.json" \
    >"$run/reward.log" 2>&1 </dev/null 9>&- &
  echo $! >"$run/reward.pid"
fi
reward_ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8033/health >"$run/reward_health.json" 2>/dev/null; then
    reward_ready=1; break
  fi
  if test -s "$run/reward.pid" && ! kill -0 "$(cat "$run/reward.pid")" 2>/dev/null; then
    tail -30 "$run/reward.log" >&2; exit 1
  fi
  sleep 2
done
test "$reward_ready" -eq 1
"$base/activevln_train_env/bin/python" - "$run/reward_health.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='ok' and x['variant']=='qwen3_exact_start_group_rank_v1'
assert x['manifest_sha256']=='1effbefb1ac9f55edf7470f9d9c83c2b4875fea71daa7199786f99a4c5ea76e1'
PY

if ! curl -fsS --max-time 2 http://127.0.0.1:5033/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/dev/shm/td_srv_qwen_confident_5033 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5033 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null 9>&- &
  echo $! >"$run/habitat.pid"
fi
habitat_ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5033/health >"$run/habitat_health.json" 2>/dev/null; then
    habitat_ready=1; break
  fi
  if test -s "$run/habitat.pid" && ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2; exit 1
  fi
  sleep 2
done
test "$habitat_ready" -eq 1
echo 'isolated confidence Qwen and Habitat services healthy'
