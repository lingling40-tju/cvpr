#!/usr/bin/env bash
set -euo pipefail

mode=${1:?branch, recovery, or counterfactual required}
steps=${2:-128}
case "$mode" in branch|recovery|counterfactual) ;; *) echo "unknown mode: $mode" >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
train="$root/runlogs/three_direction_scale_${mode}_${steps}step_suite"
run_dir="$root/runlogs/three_direction_scale_${mode}_${steps}step_eval"
result="$root/runlogs/three_direction_val256"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'scaled evaluation already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$train/suite.completed"; do
  if test -f "$train/suite.failed"; then echo 'scaled training failed' >&2; exit 1; fi
  pid=$(cat "$train/suite.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "scaled training PID $pid stopped" >&2; exit 1; fi
  sleep 60
done
if test -f "$train/no_matched_gain"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi

for seed in 11 22 33; do
  for arm in "$mode" "${mode}_control"; do
    experiment="three_directions_${arm}_${steps}step"
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    checkpoint="$root/verl_checkpoints/$experiment/global_step_${steps}/actor/huggingface"
    label="${arm}${steps}_seed${seed}"
    test -f "$root/runlogs/$experiment/completed"
    test -s "$root/runlogs/$experiment/validation.json"
    test -f "$checkpoint/config.json"
    bash tools/run_direction_eval.sh "$label" "$checkpoint" 1 2 \
      >"$run_dir/$label.launcher.log" 2>&1
    test -f "$result/$label.completed"
  done
done
"$base/activevln_server_env/bin/python" tools/analyze_scaled_val256.py \
  "$mode" --steps "$steps" >"$run_dir/analysis.log" 2>&1
test -s "$result/scale_${mode}_${steps}_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
