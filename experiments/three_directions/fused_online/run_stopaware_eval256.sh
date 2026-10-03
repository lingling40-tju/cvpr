#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
old_root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/stopaware_eval256"
result="$root/runlogs/stopaware_val256"
candidate=stopaware_group4_64_seed11
control=group4_64_seed11_fresh256
mkdir -p "$run"
exec 9>"$run/eval.lock"
flock -n 9 || { echo 'stop-aware evaluation already running' >&2; exit 2; }
if test -f "$run/eval.completed"; then exit 0; fi
rm -f "$run/eval.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/eval.failed"; fi
}
trap on_exit EXIT
test -f "$root/runlogs/stopaware_followup/training_audited"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375
candidate_checkpoint="$root/verl_checkpoints/stopaware_group4_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$source_root/verl_checkpoints/three_directions_group4_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"

# Free both GPU pairs for simultaneous candidate/control inference.
stop_service() {
  local name=$1 port=$2 expected=$3 pid_file=$4 pid
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F "$expected" >/dev/null || {
      echo "refusing to stop unmatched $name PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for attempt in $(seq 1 60); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "$name port $port did not stop" >&2
  return 1
}
stop_service stopaware_habitat 5021 server.port=5021 \
  "$root/runlogs/stopaware_services/habitat.pid"
stop_service frozen_reward 8024 failure_only_reward_server.py \
  "$old_root/runlogs/failure_only_service/service.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services.stopped"

eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    bash "$source_root/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.launcher.log" 2>&1
}
eval_lane "$candidate" "$candidate_checkpoint" 3 2 8030 &
pid_candidate=$!
eval_lane "$control" "$control_checkpoint" 1 0 8031 &
pid_control=$!
status=0
wait "$pid_candidate" || status=1
wait "$pid_control" || status=1
test "$status" -eq 0
test -f "$result/$candidate.completed"
test -f "$result/$control.completed"
"$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$control" \
  --expected-count 256 --output "$run/paired_stopaware_vs_group4.json" \
  >"$run/analysis.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_stopaware_vs_group4.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['episodes']==256 and x['scenes']==10
assert x['manifest_sha256']=='2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval.completed"
