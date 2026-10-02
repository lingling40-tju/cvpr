#!/usr/bin/env bash
set -euo pipefail

# Spend the three-seed/full-validation budget only after a positive matched
# 256-episode screen. Reuse all completed outcome-only control evaluations.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
pilot="$root/runlogs/fused_followup_watcher"
screen="$root/runlogs/fused_reward_group4_eval256"
run="$root/runlogs/fused_scale_conditional"
result="$source_root/runlogs/three_direction_full_val_unseen"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'fused scale watcher already running' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.started"

until test -f "$pilot/watcher.completed"; do
  if test -f "$pilot/watcher.failed"; then echo 'fused pilot follow-up failed' >&2; exit 1; fi
  pid=$(cat "$root/runlogs/fused_followup_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "pilot watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -f "$screen/eval.completed"
test "$(sha256sum "$source_root/runlogs/three_direction_val256/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
decision=$("$base/activevln_server_env/bin/python" - "$screen/paired_fused_vs_group4.json" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
assert p['split'] == 'val_unseen' and p['episodes'] == 256 and p['scenes'] == 11
assert p['manifest_sha256'] == '546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46'
assert p['candidate'] == 'fused_group4_64_seed11' and p['control'] == 'group4_64_seed11'
assert p['candidate_metrics']['count'] == p['control_metrics']['count'] == 256
assert p['candidate_metrics']['inference_errors'] == p['control_metrics']['inference_errors'] == 0
sr, spl = p['paired']['sr_pp'], p['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr > 0 and spl > 0 else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/pilot_decision.txt"
if [ "$decision" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test "$decision" = eligible

bash "$root/tools/start_fused_services.sh" >"$run/services.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_fused_scale_training.sh" "$seed" \
    >"$run/train_seed${seed}.launcher.log" 2>&1
  name="fused_reward_group4_128step_seed${seed}"
  test -f "$root/runlogs/$name/completed"
  test -s "$root/runlogs/$name/paired_train_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"

# Release the two frozen online services so full evaluation can use both GPU
# pairs: (model 1, simulator 0) and (model 3, simulator 2).
stop_service() {
  local name=$1 port=$2 expected=$3 pid_file="$root/runlogs/fused_services/$1.pid" pid
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
stop_service reward 8021 reward_server.py
stop_service habitat 5021 server.port=5021
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services.stopped"

test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
eval_lane() {
  local model_gpu=$1 sim_gpu=$2 port=$3 seed label control name checkpoint pair
  shift 3
  for seed in "$@"; do
    label="fused_group4_128_seed${seed}"
    control="group4_128_seed${seed}"
    name="fused_reward_group4_128step_seed${seed}"
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
eval_lane 1 0 8022 11 33 &
lane_a=$!
eval_lane 3 2 8023 22 &
lane_b=$!
status=0
wait "$lane_a" || status=1
wait "$lane_b" || status=1
test "$status" -eq 0
"$base/activevln_server_env/bin/python" "$root/tools/analyze_fused_scale.py" \
  --root "$result" --output "$run/full1839_analysis.json" \
  >"$run/aggregate.log" 2>&1
test -s "$run/full1839_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
