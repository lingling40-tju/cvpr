#!/usr/bin/env bash
set -euo pipefail

# A second stochastic-decode pass is conditional on a positive first full
# comparison. The first pass and its analysis remain untouched.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
progress_root="$base/ActiveVLN_progress_fallback_20261002"
first_suite="$root/runlogs/three_direction_progress_scale_conditional"
first_result="$root/runlogs/three_direction_full_val_unseen"
result="$root/runlogs/three_direction_full_val_unseen_progress_replication_seed20261003"
run_dir="$root/runlogs/three_direction_progress_full_replication"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'full replication already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$first_suite/suite.completed"; do
  if test -f "$first_suite/suite.failed"; then echo 'first full evaluation failed' >&2; exit 1; fi
  first_pid=$(cat "$first_suite/suite.pid")
  if ! kill -0 "$first_pid" 2>/dev/null; then
    echo "first full-evaluation PID $first_pid stopped" >&2
    exit 1
  fi
  sleep 60
done

if test -f "$first_suite/no_pilot_gain" || test -f "$first_suite/branch_gain_retained"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi

eligibility=$("$base/activevln_server_env/bin/python" - "$first_result/scale_progress_128_analysis.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data['split'] == 'val_unseen' and data['episodes'] == 1839
assert data['mode'] == 'progress' and data['train_steps'] == 128
expected = {f'{arm}128_seed{seed}' for arm in ('progress', 'branch_control')
            for seed in (11, 22, 33)}
assert set(data['models']) == expected
assert set(data['paired_seed_differences']) == {'11', '22', '33'}
assert all(model['count'] == 1839 and model['inference_errors'] == 0
           for model in data['models'].values())
print('eligible' if data['mean_paired_sr_pp'] > 0 and data['mean_paired_spl_pp'] >= 0 else 'ineligible')
PY
)
if [ "$eligibility" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$eligibility" = eligible

mkdir -p "$result"
if test ! -f "$result/manifest.json"; then
  cp "$first_result/manifest.json" "$result/manifest.json"
fi
cmp -s "$first_result/manifest.json" "$result/manifest.json"
[ "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e ]
printf '%s\n' 20261003 >"$run_dir/vllm_seed.txt"

for port in 8011 8012; do
  for attempt in $(seq 1 60); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "port $port still in use after first evaluation" >&2
    exit 1
  fi
done

evaluate_lane() {
  local arm=$1 gpu=$2 port=$3 seed label experiment checkpoint model_root
  if [ "$arm" = progress ]; then
    model_root="$progress_root"
    experiment=three_directions_progress_fallback_128step
  else
    model_root="$root"
    experiment=three_directions_branch_control_128step
  fi
  for seed in 11 22 33; do
    label="${arm}128_seed${seed}"
    local seed_experiment="$experiment"
    if [ "$seed" != 11 ]; then seed_experiment="${experiment}_seed${seed}"; fi
    checkpoint="$model_root/verl_checkpoints/$seed_experiment/global_step_128/actor/huggingface"
    test -f "$model_root/runlogs/$seed_experiment/completed"
    test -f "$checkpoint/config.json"
    VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT=1839 \
      VLN_EVAL_PORT="$port" VLN_VLLM_SEED=20261003 \
      bash tools/run_direction_eval.sh "$label" "$checkpoint" "$gpu" 2 \
      >"$run_dir/$label.launcher.log" 2>&1
    test -f "$result/$label.completed"
  done
}

evaluate_lane progress 0 8011 &
progress_pid=$!
evaluate_lane branch_control 1 8012 &
control_pid=$!
status=0
wait "$progress_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0

"$base/activevln_server_env/bin/python" tools/analyze_progress_scaled.py \
  --root "$result" --expected-count 1839 \
  >"$run_dir/analysis.log" 2>&1
test -s "$result/scale_progress_128_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
