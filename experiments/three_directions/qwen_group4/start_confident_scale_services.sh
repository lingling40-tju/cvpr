#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_qwen_confident_20261004"
source_root="$base/ActiveVLN_three_directions_20261002"
qroot="$source_root/runlogs/ordinal_progress/qwen3_route_match"
run="$root/runlogs/confident_scale_services"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'confidence scale service startup already running' >&2; exit 2; }
test -f "$root/runlogs/confident_scale/pilot_eligible"
test "$(sha256sum "$root/data/qwen3_group4_exact512.parquet" | awk '{print $1}')" = \
  d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f

if ! curl -fsS --max-time 2 http://127.0.0.1:8034/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_semantic_verifier_env/bin/python" -u \
    tools/qwen_group_rank_server.py --port 8034 \
    --manifest "$qroot/online_exact512_manifest.json" \
    --pilot-parquet "$root/data/qwen3_group4_exact512.parquet" \
    --train-dataset "$base/ActiveVLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
    --model "$base/models/Qwen3-VL-8B-Instruct" \
    --model-hashes "$qroot/model_sha256.txt" \
    --expert-analysis "$qroot/expert_calibration_analysis.json" \
    >"$run/reward.log" 2>&1 </dev/null 9>&- &
  echo $! >"$run/reward.pid"
fi
reward_ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8034/health >"$run/reward_health.json" 2>/dev/null; then
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
assert x['manifest_sha256']=='1efa550bdd21ff947ff1dfd7e461a57fb056ac034bd8f7e801beb7064da56b0a'
assert x['expert_analysis_sha256']=='6e72f6b2a8c1b73216a1f01c70cb85fc733e592035799e3a4bbf12774edca1d4'
PY

if ! curl -fsS --max-time 2 http://127.0.0.1:5034/health >/dev/null 2>&1; then
  cd "$root"
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
    RAY_TMPDIR=/dev/shm/td_srv_qwen_confident_5034 \
    "$base/activevln_server_env/bin/python" -m vlnce_server.server \
    server.port=5034 'vlnce.gpus=[0]' \
    'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$run/habitat.log" 2>&1 </dev/null 9>&- &
  echo $! >"$run/habitat.pid"
fi
habitat_ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:5034/health >"$run/habitat_health.json" 2>/dev/null; then
    habitat_ready=1; break
  fi
  if test -s "$run/habitat.pid" && ! kill -0 "$(cat "$run/habitat.pid")" 2>/dev/null; then
    tail -30 "$run/habitat.log" >&2; exit 1
  fi
  sleep 2
done
test "$habitat_ready" -eq 1
echo 'isolated confidence scale Qwen and Habitat services healthy'
