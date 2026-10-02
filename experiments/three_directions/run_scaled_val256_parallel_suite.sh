#!/usr/bin/env bash
set -euo pipefail

# Replacement watcher for the queued six-model, 256-episode screen. It does
# not modify the original waiting script or any running trainer.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
training="$root/runlogs/three_direction_scale_branch_128step_suite"
audits="$root/runlogs/three_direction_scale_branch_128step_pair_audit"
run_dir="$root/runlogs/three_direction_scale_branch_128step_eval"
result="$root/runlogs/three_direction_val256"
mkdir -p "$run_dir"
exec 9>"$run_dir/parallel.lock"
flock -n 9 || { echo 'parallel screen already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" >"$run_dir/parallel.failed"
    printf '%s\n' "$status" >"$run_dir/suite.failed"
  fi
}
trap on_exit EXIT
rm -f "$run_dir/parallel.failed" "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/parallel.started"

until test -f "$training/suite.completed"; do
  if test -f "$training/suite.failed"; then echo 'scaled training failed' >&2; exit 1; fi
  train_pid=$(cat "$training/suite.pid")
  if ! kill -0 "$train_pid" 2>/dev/null; then echo "train PID $train_pid stopped" >&2; exit 1; fi
  sleep 60
done
until test -f "$audits/watcher.completed"; do
  if test -f "$audits/watcher.failed"; then echo 'paired audit failed' >&2; exit 1; fi
  audit_pid=$(cat "$audits/watcher.pid")
  if ! kill -0 "$audit_pid" 2>/dev/null; then echo "audit PID $audit_pid stopped" >&2; exit 1; fi
  sleep 60
done

evaluate_lane() {
  local arm=$1 gpu=$2 port=$3 seed label experiment checkpoint
  for seed in 11 22 33; do
    label="${arm}128_seed${seed}"
    experiment="three_directions_${arm}_128step"
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
    test -f "$root/runlogs/$experiment/completed"
    test -s "$root/runlogs/$experiment/validation.json"
    test -f "$checkpoint/config.json"
    VLN_EVAL_PORT="$port" bash tools/run_direction_eval.sh \
      "$label" "$checkpoint" "$gpu" 2 >"$run_dir/$label.parallel_launcher.log" 2>&1
    test -f "$result/$label.completed"
  done
}

evaluate_lane branch 1 8011 &
branch_pid=$!
evaluate_lane branch_control 3 8012 &
control_pid=$!
status=0
wait "$branch_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0

"$base/activevln_server_env/bin/python" tools/analyze_scaled_val256.py \
  branch --steps 128 >"$run_dir/parallel_analysis.log" 2>&1
test -s "$result/scale_branch_128_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/parallel.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
