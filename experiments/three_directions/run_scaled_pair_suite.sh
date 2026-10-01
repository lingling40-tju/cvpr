#!/usr/bin/env bash
set -euo pipefail

mode=${1:?branch, recovery, or counterfactual required}
steps=${2:-128}
case "$mode" in branch|recovery|counterfactual) ;; *) echo "unknown mode: $mode" >&2; exit 2 ;; esac
[[ "$steps" =~ ^[0-9]+$ ]] && [ "$steps" -ge 128 ]
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
matched="$root/runlogs/three_direction_matched_followup"
finalizer="$root/runlogs/three_direction_pilot_recovery"
targeted="$root/runlogs/three_direction_val_counterfactual"
run_dir="$root/runlogs/three_direction_scale_${mode}_${steps}step_suite"
dataset="data/${mode}_scale512_train.parquet"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'scaled suite already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"
test -f "$dataset"

until test -f "$finalizer/finalizer.completed"; do
  if test -f "$finalizer/finalizer.failed"; then echo 'pilot finalizer failed' >&2; exit 1; fi
  pid=$(cat "$finalizer/finalizer.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "finalizer PID $pid stopped" >&2; exit 1; fi
  sleep 60
done
until test -f "$matched/suite.completed"; do
  if test -f "$matched/suite.failed"; then echo 'matched suite failed' >&2; exit 1; fi
  pid=$(cat "$matched/suite.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "matched PID $pid stopped" >&2; exit 1; fi
  sleep 60
done
eligibility=$("$base/activevln_server_env/bin/python" - "$mode" <<'PY'
import json, sys
from pathlib import Path
mode = sys.argv[1]
path = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val256/matched_analysis.json')
data = json.loads(path.read_text())
pair = data['paired_vs_matched_control'].get(mode)
print('eligible' if pair and pair['sr_pp'] > 0 and pair['spl_pp'] >= 0 else 'ineligible')
PY
)
if [ "$eligibility" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_matched_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$eligibility" = eligible

# The targeted counterfactual diagnostic uses GPUs 1/2 and the evaluator
# port. Let it finish before reserving all four GPUs for scaled training.
until test -f "$targeted/suite.completed"; do
  if test -f "$targeted/suite.failed"; then echo 'targeted evaluation failed' >&2; exit 1; fi
  pid=$(cat "$targeted/suite.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "targeted PID $pid stopped" >&2; exit 1; fi
  sleep 60
done

for seed in 11 22 33; do
  pids=()
  arms=()
  for arm in "$mode" "${mode}_control"; do
    experiment="three_directions_${arm}_${steps}step"
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    train_dir="$root/runlogs/$experiment"
    checkpoint="$root/verl_checkpoints/$experiment/global_step_${steps}/actor/huggingface"
    if test -f "$train_dir/completed"; then
      test -s "$train_dir/validation.json"
      test -f "$checkpoint/config.json"
      continue
    fi
    if [ "$arm" = "$mode" ]; then
      gpus=0,1
      port=5002
    else
      gpus=2,3
      port=5007
    fi
    curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null
    env VLN_TRAIN_DATASET="$dataset" VLN_PILOT_SERVICE_URL="http://127.0.0.1:$port" \
      bash tools/run_direction_pilot.sh "$arm" "$steps" "$gpus" "$seed" \
      >"$run_dir/${arm}_seed${seed}.launcher.log" 2>&1 &
    child=$!
    pids+=("$child")
    arms+=("$arm")
    echo "$child" >"$run_dir/${arm}_seed${seed}.pid"
  done
  failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=1
    arm=${arms[$index]}
    experiment="three_directions_${arm}_${steps}step"
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    test -f "$root/runlogs/$experiment/completed" || failed=1
  done
  test "$failed" -eq 0
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
