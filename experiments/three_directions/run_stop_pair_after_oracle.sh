#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stop_pair_group4_20261004"
oracle="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
prior="$oracle/runlogs/oracle_pilot"
run="$root/runlogs/stop_pair_after_oracle"
result="$root/runlogs/stop_pair_val256"
mkdir -p "$run" "$result"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'STOP-pair watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
service_started=0
stop_service() {
  local pid_file="$root/runlogs/stop_pair_service/habitat.pid" pid
  if test "$service_started" -ne 1 || ! test -s "$pid_file"; then return 0; fi
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F 'server.port=5036' >/dev/null || {
      echo "refusing to stop unmatched PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5036/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo 'STOP-pair Habitat port 5036 did not stop' >&2
  return 1
}
on_exit() {
  local status=$?
  stop_service || status=1
  if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$prior/completed"; do
  if test -f "$prior/failed"; then
    echo 'oracle pilot failed; repair before conditional STOP-pair pilot' >&2
    exit 1
  fi
  if test -s "$prior/launcher.pid" && \
      ! kill -0 "$(cat "$prior/launcher.pid")" 2>/dev/null; then
    echo 'oracle launcher disappeared without a terminal marker' >&2
    exit 1
  fi
  sleep 30
done
if test -f "$prior/positive_upperbound"; then
  test ! -f "$prior/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_oracle_positive"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
test -f "$prior/no_pilot_gain"
test -s "$prior/paired_oracle_vs_control.json"
test -f "$oracle/runlogs/oracle_val256/oracle_turnwise_64_seed11.completed"
test -f "$oracle/runlogs/oracle_val256/qwen3_exact_control_64_seed11_oracle_screen.completed"
for port in 5035 8091 8092 5036 8093 8094; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || \
      curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "required port $port still in use" >&2; exit 1
  fi
done
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[2]}" -lt 8000 && \
      test "${used[3]}" -lt 8000; then break; fi
  sleep 30
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/gpus_released"
manifest="$oracle/tools/process_val256_manifest.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
cp "$manifest" "$result/manifest.json"

service_started=1
bash "$root/tools/start_stop_pair_service.sh" >"$run/service_launcher.log" 2>&1
bash "$root/tools/run_stop_pair_train.sh" 2 >"$run/train_2step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" "$root/tools/audit_stop_pair_train.py" \
  --root "$root" --control-root "$control" --steps 2 \
  --output "$run/train_audit_2step.json" >"$run/train_audit_2step.log" 2>&1
test -f "$root/runlogs/stop_pair_2step_seed11/audited"
test -s "$run/train_audit_2step.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/smoke_passed"

bash "$root/tools/run_stop_pair_train.sh" 64 >"$run/train_64step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" "$root/tools/audit_stop_pair_train.py" \
  --root "$root" --control-root "$control" --steps 64 \
  --output "$run/train_audit_64step.json" >"$run/train_audit_64step.log" 2>&1
test -s "$run/train_audit_64step.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_64step_audited"
stop_service
service_started=0
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[1]}" -lt 8000 && \
      test "${used[2]}" -lt 8000 && \
      test "${used[3]}" -lt 8000; then break; fi
  sleep 10
done

candidate=stop_pair_64_seed11
baseline=qwen3_exact_control_64_seed11_stop_pair_screen
candidate_checkpoint="$root/verl_checkpoints/stop_pair_64step_seed11/global_step_64/actor/huggingface"
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
eval_lane "$candidate" "$candidate_checkpoint" 3 2 8093 & candidate_pid=$!
eval_lane "$baseline" "$control_checkpoint" 1 0 8094 & control_pid=$!
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0
test -f "$result/$candidate.completed"
test -f "$result/$baseline.completed"
"$base/activevln_server_env/bin/python" "$control/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_stop_pair_vs_control.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" "$control/tools/export_matched_episodes.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_stop_pair_episodes.jsonl" \
  >"$run/paired_export.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_stop_pair_vs_control.json" "$run" <<'PY'
import json,math,pathlib,sys
x=json.load(open(sys.argv[1]));run=pathlib.Path(sys.argv[2])
assert x['split']=='val_unseen' and x['episodes']==256
assert x['manifest_sha256']=='bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c'
assert x['candidate']=='stop_pair_64_seed11'
assert x['control']=='qwen3_exact_control_64_seed11_stop_pair_screen'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert math.isfinite(x['paired']['sr_pp']) and math.isfinite(x['paired']['spl_pp'])
name='positive_upperbound' if x['paired']['sr_pp']>0 and x['paired']['spl_pp']>0 else 'no_pilot_gain'
(run/name).write_text('privileged simulator-distance STOP diagnostic only\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
