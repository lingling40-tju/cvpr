#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
train_dir="$root/runlogs/three_direction_pilot_suite"
eval_dir="$root/runlogs/three_direction_val256"
mkdir -p "$eval_dir"
exec 9>"$eval_dir/suite.lock"
flock -n 9 || { echo 'evaluation suite already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$eval_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$eval_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$eval_dir/suite.started"

until test -f "$train_dir/suite.completed"; do
  if test -f "$train_dir/suite.failed"; then
    echo 'training suite failed; evaluation cannot start' >&2
    exit 1
  fi
  train_pid=$(cat "$train_dir/suite.pid")
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "training suite PID $train_pid stopped without completion" >&2
    exit 1
  fi
  sleep 60
done

for mode in branch recovery counterfactual; do
  label="${mode}64"
  checkpoint="$root/verl_checkpoints/three_directions_${mode}_64step/global_step_64/actor/huggingface"
  test -f "$root/runlogs/three_directions_${mode}_64step/completed"
  test -f "$checkpoint/config.json"
  bash tools/run_direction_eval.sh "$label" "$checkpoint" 1 2 \
    >"$eval_dir/${label}.launcher.log" 2>&1
  test -f "$eval_dir/$label.completed"
done

"/Knowin/foundation/haozhiwang/whz/activevln_server_env/bin/python" \
  tools/analyze_direction_eval.py branch64 recovery64 counterfactual64 \
  >"$eval_dir/analysis.log" 2>&1
test -s "$eval_dir/analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$eval_dir/suite.completed"
