#!/usr/bin/env bash
set -euo pipefail

# Full-budget work is conditional on a complete matched, disjoint 256-episode
# pilot. All training groups stay at four; cached controls save evaluation GPU.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
pilot="$root/runlogs/stopaware_followup"
screen="$root/runlogs/stopaware_eval256"
run="$root/runlogs/stopaware_scale"
result="$source_root/runlogs/three_direction_full_val_unseen"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'stop-aware scale watcher already running' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.started"

until test -f "$pilot/followup.completed"; do
  if test -f "$pilot/followup.failed"; then echo 'stop-aware pilot failed' >&2; exit 1; fi
  pid=$(cat "$pilot/launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "pilot watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -f "$screen/eval.completed"
decision=$("$base/activevln_server_env/bin/python" - "$screen/paired_stopaware_vs_group4.json" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
assert p['split'] == 'val_unseen' and p['episodes'] == 256 and p['scenes'] == 10
assert p['manifest_sha256'] == '2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375'
assert p['candidate'] == 'stopaware_group4_64_seed11'
assert p['control'] == 'group4_64_seed11_fresh256'
assert p['candidate_metrics']['count'] == p['control_metrics']['count'] == 256
assert p['candidate_metrics']['inference_errors'] == p['control_metrics']['inference_errors'] == 0
sr, spl = p['paired']['sr_pp'], p['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr > 0 and spl > 0 else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/pilot_decision.txt"
if test "$decision" = ineligible; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test "$decision" = eligible
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/pilot_eligible"

bash "$root/tools/start_failure_only_service.sh" >"$run/services.log" 2>&1
bash "$root/tools/start_stopaware_habitat.sh" >>"$run/services.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_stopaware_scale_training.sh" 128 "$seed" \
    >"$run/train_seed${seed}.launcher.log" 2>&1
  name="stopaware_group4_128step_seed${seed}"
  test -f "$root/runlogs/$name/completed"
  python3 "$root/tools/audit_stopaware_scale.py" \
    --root "$root" --source-root "$source_root" --steps 128 --seed "$seed" \
    >"$root/runlogs/$name/paired_train_audit.json"
  test -s "$root/runlogs/$name/paired_train_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"

# Stop frozen services so the four GPUs can host two independent eval lanes.
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
  "$base/ActiveVLN_fused_reward_20261003/runlogs/failure_only_service/service.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services.stopped"

test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
eval_lane() {
  local model_gpu=$1 sim_gpu=$2 port=$3 seed label control name checkpoint pair
  shift 3
  for seed in "$@"; do
    label="stopaware_group4_128_seed${seed}"
    control="group4_128_seed${seed}"
    name="stopaware_group4_128step_seed${seed}"
    checkpoint="$root/verl_checkpoints/$name/global_step_128/actor/huggingface"
    test -f "$root/runlogs/$name/completed"
    test -f "$result/$control.completed"
    test -f "$checkpoint/config.json"
    VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT=1839 VLN_EVAL_PORT="$port" \
      VLN_EVAL_SHARDS=4 bash "$source_root/tools/run_direction_eval.sh" \
        "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
        >"$run/eval_seed${seed}.launcher.log" 2>&1
    test -f "$result/$label.completed"
    pair="$result/paired_${label}_vs_${control}.json"
    "$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
      --root "$result" --candidate "$label" --control "$control" \
      --expected-count 1839 --output "$pair" \
      >"$run/analyze_seed${seed}.log" 2>&1
    test -s "$pair"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval_seed${seed}.completed"
  done
}
eval_lane 1 0 8032 11 33 &
lane_a=$!
eval_lane 3 2 8033 22 &
lane_b=$!
status=0
wait "$lane_a" || status=1
wait "$lane_b" || status=1
test "$status" -eq 0
"$base/activevln_server_env/bin/python" "$root/tools/analyze_stopaware_scale.py" \
  --root "$result" --candidate-prefix stopaware_group4_128 \
  --output "$run/full1839_analysis.json" >"$run/aggregate.log" 2>&1
test -s "$run/full1839_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
