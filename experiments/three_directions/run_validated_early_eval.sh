#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
run_dir="$root/runlogs/three_direction_early_eval"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'early evaluation already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

for mode in branch recovery; do
  test -s "$root/runlogs/three_directions_${mode}_64step/checkpoint_validated.json"
  model="$root/verl_checkpoints/three_directions_${mode}_64step/global_step_64/actor/huggingface"
  label="${mode}64"
  # GPU2 is free for the model; GPU3 shares Habitat with a small simulator
  # service. The counterfactual trainer occupies GPUs 0 and 1.
  bash tools/run_direction_eval.sh "$label" "$model" 2 3 \
    >"$run_dir/${label}.launcher.log" 2>&1
  test -f "$root/runlogs/three_direction_val256/$label.completed"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
