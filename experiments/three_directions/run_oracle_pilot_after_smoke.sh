#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
smoke="$root/runlogs/oracle_after_recheck"
run="$root/runlogs/oracle_pilot"
result="$root/runlogs/oracle_val256"
mkdir -p "$run" "$result"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'oracle pilot watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
service_started=0
stop_service() {
  local pid_file="$root/runlogs/oracle_turnwise_service/habitat.pid" pid
  if test "$service_started" -ne 1 || ! test -s "$pid_file"; then return 0; fi
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F 'server.port=5035' >/dev/null || {
      echo "refusing to stop unmatched PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5035/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo 'oracle Habitat port 5035 did not stop' >&2
  return 1
}
on_exit() {
  local status=$?
  stop_service || status=1
  if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$smoke/completed"; do
  if test -f "$smoke/failed"; then
    echo 'oracle two-step smoke failed; refusing the 64-step pilot' >&2
    exit 1
  fi
  if test -s "$smoke/launcher.pid" && \
      ! kill -0 "$(cat "$smoke/launcher.pid")" 2>/dev/null; then
    echo 'oracle smoke launcher disappeared without terminal marker' >&2
    exit 1
  fi
  sleep 30
done
test -f "$root/runlogs/oracle_turnwise_2step_seed11/audited"
test -s "$smoke/train_audit_2step.json"
test -f "$control/runlogs/qwen3_exact_control_64step_seed11/completed"
test -f "$control/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface/config.json"
manifest="$root/tools/process_val256_manifest.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
cp "$manifest" "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c

service_started=1
bash "$root/tools/start_oracle_turnwise_service.sh" \
  >"$run/service_launcher.log" 2>&1
bash "$root/tools/run_oracle_turnwise_train.sh" 64 \
  >"$run/train_64step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" \
  "$root/tools/audit_oracle_turnwise_train.py" \
  --root "$root" --control-root "$control" --steps 64 \
  --output "$run/train_audit_64step.json" \
  >"$run/train_audit_64step.log" 2>&1
test -s "$run/train_audit_64step.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_64step_audited"
stop_service
service_started=0
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[2]}" -lt 8000 && \
      test "${used[3]}" -lt 8000; then
    break
  fi
  sleep 10
done

candidate=oracle_turnwise_64_seed11
baseline=qwen3_exact_control_64_seed11_oracle_screen
candidate_checkpoint="$root/verl_checkpoints/oracle_turnwise_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$control/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"
eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED=11 \
    bash "$control/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
eval_lane "$candidate" "$candidate_checkpoint" 3 2 8091 & candidate_pid=$!
eval_lane "$baseline" "$control_checkpoint" 1 0 8092 & control_pid=$!
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0
test -f "$result/$candidate.completed"
test -f "$result/$baseline.completed"
"$base/activevln_server_env/bin/python" \
  "$control/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_oracle_vs_control.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" \
  "$control/tools/export_matched_episodes.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_oracle_episodes.jsonl" \
  >"$run/paired_export.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_oracle_vs_control.json" <<'PY'
import json, math, sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256
assert x['manifest_sha256']=='bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c'
assert x['candidate']=='oracle_turnwise_64_seed11'
assert x['control']=='qwen3_exact_control_64_seed11_oracle_screen'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert math.isfinite(x['paired']['sr_pp']) and math.isfinite(x['paired']['spl_pp'])
PY
"$base/activevln_server_env/bin/python" - "$run/paired_oracle_vs_control.json" "$run" <<'PY'
import json, pathlib, sys
x=json.load(open(sys.argv[1]))
run=pathlib.Path(sys.argv[2])
name='positive_upperbound' if x['paired']['sr_pp']>0 and x['paired']['spl_pp']>0 else 'no_pilot_gain'
(run/name).write_text('privileged simulator-distance diagnostic only\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
