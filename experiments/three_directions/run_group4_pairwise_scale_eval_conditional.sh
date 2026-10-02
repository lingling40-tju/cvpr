#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_group4_pairwise_20261002"
source_root="$base/ActiveVLN_three_directions_20261002"
gate="$root/runlogs/group4_pairwise_scale_conditional"
group4_eval="$source_root/runlogs/optimizer_scale_eval_conditional"
run_dir="$root/runlogs/group4_pairwise_scale_eval_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'pairwise scale evaluator already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$gate/suite.completed"; do
  if test -f "$gate/suite.failed"; then echo 'pairwise scale training failed' >&2; exit 1; fi
  pid_file="$root/runlogs/group4_pairwise_scale_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "pairwise scale PID $pid stopped" >&2; exit 1; }
  sleep 60
done
decision=$(cat "$gate/pilot_decision.txt")
if [ "$decision" = ineligible ]; then
  test -f "$gate/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_eligible_pilot"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$decision" = eligible
test -f "$gate/training.completed"
test -f "$gate/service.stopped"
for seed in 11 22 33; do test -f "$gate/train_seed${seed}.completed"; done

for count in 256 1839; do
  if [ "$count" = 256 ]; then
    result="$source_root/runlogs/three_direction_val256"
    expected_sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
  else
    result="$source_root/runlogs/three_direction_full_val_unseen"
    expected_sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$expected_sha"
  for seed in 11 22 33; do
    label="group4_pairwise_128_seed${seed}"
    control="branch_control128_seed${seed}"
    experiment="three_directions_group4_pairwise_128step_seed${seed}"
    checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
    test -f "$root/runlogs/$experiment/completed"
    test -f "$checkpoint/config.json"
    test -f "$result/$control.completed"
    VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT="$count" VLN_EVAL_PORT=8020 \
      bash "$source_root/tools/run_direction_eval.sh" "$label" "$checkpoint" 1 0 \
      >"$run_dir/eval_${count}_seed${seed}.launcher.log" 2>&1
    test -f "$result/$label.completed"
    pair="$result/paired_${label}_vs_${control}.json"
    "$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
      --root "$result" --candidate "$label" --control "$control" \
      --expected-count "$count" --output "$pair" \
      >"$run_dir/analyze_control_${count}_seed${seed}.log" 2>&1
    test -s "$pair"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/eval_${count}_seed${seed}.completed"
  done
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/candidate_evaluations.completed"

until test -f "$group4_eval/suite.completed"; do
  if test -f "$group4_eval/suite.failed"; then echo 'four-way evaluation failed' >&2; exit 1; fi
  pid_file="$source_root/runlogs/optimizer_scale_eval_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "four-way evaluation PID $pid stopped" >&2; exit 1; }
  sleep 60
done

for count in 256 1839; do
  if [ "$count" = 256 ]; then
    result="$source_root/runlogs/three_direction_val256"
  else
    result="$source_root/runlogs/three_direction_full_val_unseen"
  fi
  for seed in 11 22 33; do
    label="group4_pairwise_128_seed${seed}"
    control="group4_128_seed${seed}"
    test -f "$result/$control.completed"
    pair="$result/paired_${label}_vs_${control}.json"
    "$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
      --root "$result" --candidate "$label" --control "$control" \
      --expected-count "$count" --output "$pair" \
      >"$run_dir/analyze_group4_${count}_seed${seed}.log" 2>&1
    test -s "$pair"
  done
  "$base/activevln_server_env/bin/python" "$root/tools/analyze_group4_pairwise_scaled.py" \
    --root "$result" --count "$count" \
    --output "$result/scale_group4_pairwise_128_analysis.json" \
    >"$run_dir/aggregate_${count}.log" 2>&1
  test -s "$result/scale_group4_pairwise_128_analysis.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/aggregate_${count}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
