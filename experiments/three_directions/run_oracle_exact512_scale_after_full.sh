#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
progress="$base/policy_progress_lora_20261004"
full="$root/runlogs/oracle_full_recheck"
run="$root/runlogs/oracle_exact512_scale"
result="$root/runlogs/oracle_exact512_full1839"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'oracle exact512 scale suite already active' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/suite.failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.started"

while ! test -f "$full/completed"; do
  if test -f "$full/failed"; then echo 'oracle full recheck failed' >&2; exit 1; fi
  if test -s "$full/launcher.pid" && \
     ! kill -0 "$(cat "$full/launcher.pid")" 2>/dev/null; then
    echo 'oracle full recheck launcher vanished' >&2; exit 1
  fi
  sleep 30
done
decision=$("$base/activevln_server_env/bin/python" - \
  "$full/oracle_full_recheck_analysis.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='oracle_turnwise_one_seed_postscreen_full_recheck_v1'
assert x['full_manifest_sha256']=='262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e'
assert x['screen_manifest_sha256']=='bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c'
assert x['candidate']=='oracle_turnwise_64_seed11_full_recheck'
assert x['control']=='qwen3_exact_control_64_seed11_oracle_full_recheck'
for name,n in [('full',1839),('outside_reused_screen',1583)]:
    scope=x[name]
    assert scope['episodes']==n
    assert scope['candidate_metrics']['count']==scope['control_metrics']['count']==n
    assert scope['candidate_metrics']['inference_errors']==scope['control_metrics']['inference_errors']==0
    assert all(math.isfinite(scope['paired'][k]) for k in ('sr_pp','spl_pp'))
print('eligible' if all(x[name]['paired'][metric]>0
                        for name in ('full','outside_reused_screen')
                        for metric in ('sr_pp','spl_pp')) else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/full_recheck_decision.txt"
if test "$decision" = ineligible; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/no_full_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test "$decision" = eligible
test "$(sha256sum "$root/data/qwen3_group4_exact512.parquet" | awk '{print $1}')" = \
  d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f
test "$(sha256sum "$control/data/qwen3_group4_exact512.parquet" | awk '{print $1}')" = \
  d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eligible"

# The independent representation fit already uses GPU 3. Do not overlap
# a four-GPU RL run with it, regardless of its eventual development gate.
while ! test -f "$progress/runlogs/full/completed"; do
  if test -f "$progress/runlogs/full/failed"; then
    echo 'progress representation fit failed; diagnose before scale' >&2; exit 1
  fi
  if test -s "$progress/runlogs/after_oracle_candidate/watcher.pid" && \
     ! kill -0 "$(cat "$progress/runlogs/after_oracle_candidate/watcher.pid")" 2>/dev/null; then
    echo 'progress representation watcher vanished' >&2; exit 1
  fi
  sleep 30
done
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[1]}" -lt 8000 && \
     test "${used[2]}" -lt 8000 && test "${used[3]}" -lt 8000; then break; fi
  sleep 20
done

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

bash "$control/tools/start_group4_service.sh" 0 >"$run/control_service.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_oracle_exact512_train.sh" control "$seed" \
    >"$run/control_seed${seed}.launcher.log" 2>&1
  test -f "$control/runlogs/oracle_exact512_control_128_seed${seed}/completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/control_seed${seed}.completed"
done
stop_service control_habitat 5013 server.port=5013 \
  "$control/runlogs/group4_service/server.pid"

bash "$root/tools/start_oracle_turnwise_service.sh" \
  >"$run/candidate_service.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_oracle_exact512_train.sh" candidate "$seed" \
    >"$run/candidate_seed${seed}.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" \
    "$root/tools/audit_oracle_exact512_scale.py" \
    --candidate-root "$root" --control-root "$control" --seed "$seed" \
    --output "$run/train_audit_seed${seed}.json" \
    >"$run/train_audit_seed${seed}.log" 2>&1
  test -s "$run/train_audit_seed${seed}.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/candidate_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"
stop_service oracle_habitat 5035 server.port=5035 \
  "$root/runlogs/oracle_turnwise_service/habitat.pid"

mkdir -p "$result"
cp "$control/runlogs/three_direction_full_val_unseen/manifest.json" \
  "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5 seed=$6
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED="$seed" bash "$control/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
for seed in 11 22 33; do
  candidate="oracle_turnwise_exact512_128_seed${seed}"
  baseline="oracle_exact512_control_128_seed${seed}"
  candidate_checkpoint="$root/verl_checkpoints/$candidate/global_step_128/actor/huggingface"
  control_checkpoint="$control/verl_checkpoints/$baseline/global_step_128/actor/huggingface"
  test -f "$candidate_checkpoint/config.json" && test -f "$control_checkpoint/config.json"
  eval_lane "$candidate" "$candidate_checkpoint" 3 2 8121 "$seed" & candidate_pid=$!
  eval_lane "$baseline" "$control_checkpoint" 1 0 8122 "$seed" & control_pid=$!
  status=0
  wait "$candidate_pid" || status=1
  wait "$control_pid" || status=1
  test "$status" -eq 0
  test -f "$result/$candidate.completed" && test -f "$result/$baseline.completed"
  "$base/activevln_server_env/bin/python" \
    "$control/tools/analyze_matched_pair.py" \
    --root "$result" --candidate "$candidate" --control "$baseline" \
    --expected-count 1839 \
    --output "$result/paired_${candidate}_vs_${baseline}.json" \
    >"$run/paired_seed${seed}.log" 2>&1
  "$base/activevln_server_env/bin/python" \
    "$control/tools/export_matched_episodes.py" \
    --root "$result" --candidate "$candidate" --control "$baseline" \
    --expected-count 1839 \
    --output "$result/paired_${candidate}_vs_${baseline}_episodes.jsonl" \
    >"$run/export_seed${seed}.log" 2>&1
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval_seed${seed}.completed"
done
PYTHONPATH="$control/tools${PYTHONPATH:+:$PYTHONPATH}" \
  "$base/activevln_server_env/bin/python" \
  "$root/tools/analyze_oracle_exact512_scale.py" \
  --root "$result" --screen-manifest "$root/tools/process_val256_manifest.json" \
  --output "$run/full1839_analysis.json" \
  >"$run/full1839_analysis.log" 2>&1
test -s "$run/full1839_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
