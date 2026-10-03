#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_qwen_group_rank_20261004"
source_root="$base/ActiveVLN_three_directions_20261002"
qroot="$source_root/runlogs/ordinal_progress/qwen3_route_match"
run="$root/runlogs/qwen_group_scale_services"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'scale service startup already running' >&2; exit 2; }
test -f "$root/runlogs/qwen_group_followup/completed"

if ! curl -fsS --max-time 2 http://127.0.0.1:8032/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_semantic_verifier_env/bin/python" -u \
    tools/qwen_group_rank_server.py --port 8032 \
    --manifest "$qroot/online_exact512_manifest.json" \
    --pilot-parquet "$root/data/qwen3_group4_exact512.parquet" \
    --train-dataset "$base/ActiveVLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
    --model "$base/models/Qwen3-VL-8B-Instruct" \
    --model-hashes "$qroot/model_sha256.txt" \
    --expert-analysis "$qroot/expert_calibration_analysis.json" \
    >"$run/reward.log" 2>&1 </dev/null &
  echo $! >"$run/reward.pid"
fi
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8032/health >"$run/reward_health.json" 2>/dev/null; then
    break
  fi
  if test -s "$run/reward.pid" && ! kill -0 "$(cat "$run/reward.pid")" 2>/dev/null; then
    tail -30 "$run/reward.log" >&2; exit 1
  fi
  sleep 2
done
"$base/activevln_train_env/bin/python" - "$run/reward_health.json" "$qroot/online_exact512_manifest.json" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1]))
expected=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
assert x['status']=='ok' and x['variant']=='qwen3_exact_start_group_rank_v1'
assert x['manifest_sha256']==expected
PY

if ! curl -fsS --max-time 2 http://127.0.0.1:5032/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/dev/shm/td_srv_qwen_group_5032 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5032 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null &
  echo $! >"$run/habitat.pid"
fi
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5032/health >"$run/habitat_health.json" 2>/dev/null; then
    break
  fi
  if test -s "$run/habitat.pid" && ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2; exit 1
  fi
  sleep 2
done
echo 'Qwen scale reward and Habitat services healthy'
