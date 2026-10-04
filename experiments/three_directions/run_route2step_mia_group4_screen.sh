#!/usr/bin/env bash
set -euo pipefail

# Use GPU 1 after the last outcome-control full evaluation, sharing the exact
# inference-GPU lock with the candidate evaluator. The 128-query cache is
# resumable; no online RL is launched here.
base=/Knowin/foundation/haozhiwang/whz
root="$base/route2step_mia_20261004"
source_root="$base/ActiveVLN_three_directions_20261002/runlogs/ordinal_progress"
scale="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_scale"
control_eval="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_control_eval_overlap"
gpu_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu1.lock"
run="$root/runlogs/screen"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'MIA screen already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$root/runlogs/download/completed"
test -f "$root/runlogs/smoke/completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  4d8678956043cddfb189c78e888fea2c487d1aa2a5ba9cdeba856f4d68b4154d
while ! test -f "$control_eval/completed"; do
  for failure in "$scale/suite.failed" "$control_eval/failed"; do
    test ! -f "$failure" || { echo "upstream failure: $failure" >&2; exit 1; }
  done
  sleep 60
done
# The same flock is held from model startup through vLLM cleanup by
# run_direction_eval.sh. If a candidate evaluation acquired it first,
# wait for that evaluation to finish before loading MIA.
exec 8>"$gpu_lock"
flock 8
used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n '2p' | tr -d ' ')
test -n "$used1" && test "$used1" -lt 8000
! curl -fsS --max-time 1 http://127.0.0.1:8122/v1/models >/dev/null 2>&1
sha256sum "$root/infer_route2step_mia_screen.py" \
  "$root/analyze_route2step_mia_screen.py" \
  "$root/smoke_route2step_mia_progress.py" \
  "$root/freeze_route2step_mia_screen.py" \
  "$run/manifest.json" "$root/model/MIA/config.json" >"$run/source.sha256"
CUDA_VISIBLE_DEVICES=1 timeout 5400 "$base/activevln_train_env/bin/python" \
  "$root/infer_route2step_mia_screen.py" \
  --model "$root/model/MIA" \
  --manifest "$run/manifest.json" \
  --record-root "$source_root/policy_process_turns" \
  --output "$run/responses.jsonl" >"$run/inference.log" 2>&1
"$base/activevln_train_env/bin/python" \
  "$root/analyze_route2step_mia_screen.py" \
  --manifest "$run/manifest.json" \
  --record-root "$source_root/policy_process_turns" \
  --responses "$run/responses.jsonl" \
  --output "$run/development.json" >"$run/analysis.log" 2>&1
test -s "$run/development.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
