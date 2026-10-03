#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_qwen_group_rank_20261004"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/qwen_group_followup"
control_run="$source_root/runlogs/qwen3_exact_control_64step_seed11"
mkdir -p "$run"
exec 9>"$run/followup.lock"
flock -n 9 || { echo 'Qwen group follow-up already running' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT

while ! test -f "$control_run/completed"; do
  if test -f "$control_run/failed"; then
    echo 'matched outcome control failed' >&2; exit 1
  fi
  if test -s "$control_run/launcher.pid" && \
      ! kill -0 "$(cat "$control_run/launcher.pid")" 2>/dev/null; then
    echo 'matched outcome control launcher exited without completion' >&2
    exit 1
  fi
  sleep 60
done
"$base/activevln_train_env/bin/python" \
  "$source_root/tools/audit_qwen_exact_control.py" \
  --root "$source_root" --output "$run/control_training_audit.json" \
  >"$run/control_audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/control_audited"

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
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "$name port $port did not stop" >&2
  return 1
}

stop_service outcome_control_habitat 5013 server.port=5013 \
  "$source_root/runlogs/group4_service/server.pid"
bash "$root/tools/start_qwen_group_services.sh" all \
  >"$run/candidate_services.log" 2>&1

bash "$root/tools/run_qwen_group_pilot.sh" 2 \
  >"$run/candidate_2step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" \
  "$root/tools/audit_qwen_group_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 2 \
  --output "$run/candidate_2step_audit.json" \
  >"$run/candidate_2step_audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/two_step_audited"

bash "$root/tools/run_qwen_group_pilot.sh" 64 \
  >"$run/candidate_64step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" \
  "$root/tools/audit_qwen_group_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 64 \
  --output "$run/candidate_64step_audit.json" \
  >"$run/candidate_64step_audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/candidate_training_audited"

stop_service candidate_habitat 5031 server.port=5031 \
  "$root/runlogs/qwen_group_services/habitat.pid"
stop_service qwen_teacher 8031 qwen_group_rank_server.py \
  "$root/runlogs/qwen_group_services/reward.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services_stopped"

result="$root/runlogs/qwen_group_val256"
mkdir -p "$result"
cp "$base/ActiveVLN_mode_rank_20261003/runlogs/mode_rank_val256/manifest.json" \
  "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc
candidate_checkpoint="$root/verl_checkpoints/qwen_group_rank_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$source_root/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"

eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED=11 \
    bash "$source_root/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
eval_lane qwen_group_rank_64_seed11 "$candidate_checkpoint" 3 2 8041 &
candidate_pid=$!
eval_lane qwen_exact_control_64_seed11 "$control_checkpoint" 1 0 8042 &
control_pid=$!
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0
test -f "$result/qwen_group_rank_64_seed11.completed"
test -f "$result/qwen_exact_control_64_seed11.completed"
"$base/activevln_server_env/bin/python" \
  "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate qwen_group_rank_64_seed11 \
  --control qwen_exact_control_64_seed11 --expected-count 256 \
  --output "$run/paired_qwen_vs_control.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_qwen_vs_control.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256 and x['scenes']==9
assert x['manifest_sha256']=='1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
