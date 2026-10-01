#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
eval_dir="$root/runlogs/three_direction_val256"
run_dir="$root/runlogs/three_direction_matched_followup"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'matched follow-up already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$eval_dir/suite.completed"; do
  if test -f "$eval_dir/suite.failed"; then
    echo 'screening evaluation failed' >&2
    exit 1
  fi
  eval_pid=$(cat "$eval_dir/suite.pid")
  if ! kill -0 "$eval_pid" 2>/dev/null; then
    echo "screening evaluation PID $eval_pid stopped without completion" >&2
    exit 1
  fi
  sleep 60
done

"$base/activevln_server_env/bin/python" - >"$run_dir/selected_modes.txt" <<'PY'
import json
from pathlib import Path
root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val256')
data = json.loads((root / 'analysis.json').read_text())
baseline = data['models']['seed11_control']
for mode in ('branch', 'recovery', 'counterfactual'):
    candidate = data['models'][f'{mode}64']
    if candidate['successes'] > baseline['successes'] and candidate['spl'] >= baseline['spl']:
        print(mode)
PY
mapfile -t selected <"$run_dir/selected_modes.txt"
if [ "${#selected[@]}" -eq 0 ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_screening_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi

# A selected matched control may have been started on GPUs 2/3 as soon as its
# pilot evaluation finished, concurrently with the remaining candidate run.
# Wait for that exact job instead of launching a duplicate into its files.
if test -f "$run_dir/branch_control.pid"; then
  branch_run="$root/runlogs/three_directions_branch_control_64step"
  branch_pid=$(cat "$run_dir/branch_control.pid")
  until test -f "$branch_run/completed"; do
    if test -f "$branch_run/failed"; then
      echo 'early branch matched control failed' >&2
      exit 1
    fi
    if ! kill -0 "$branch_pid" 2>/dev/null; then
      echo "early branch control PID $branch_pid stopped without completion" >&2
      exit 1
    fi
    sleep 60
  done
fi

pending=()
for mode in "${selected[@]}"; do
  if ! test -f "$root/runlogs/three_directions_${mode}_control_64step/completed"; then
    pending+=("$mode")
  fi
done

for ((i=0; i<${#pending[@]}; i+=2)); do
  pids=()
  modes=()
  for lane in 0 1; do
    index=$((i + lane))
    if [ "$index" -ge "${#pending[@]}" ]; then continue; fi
    mode=${pending[$index]}
    if [ "$lane" -eq 0 ]; then
      gpus=0,1
      port=5002
    else
      gpus=2,3
      port=5007
    fi
    curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null
    env VLN_PILOT_SERVICE_URL="http://127.0.0.1:$port" \
      bash tools/run_direction_pilot.sh "${mode}_control" 64 "$gpus" 11 \
      >"$run_dir/${mode}_control.launcher.log" 2>&1 &
    child_pid=$!
    pids+=("$child_pid")
    modes+=("$mode")
    echo "$child_pid" >"$run_dir/${mode}_control.pid"
  done
  failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=1
    test -f "$root/runlogs/three_directions_${modes[$index]}_control_64step/completed" || failed=1
  done
  test "$failed" -eq 0
done

labels=(branch64 recovery64 counterfactual64)
for mode in "${selected[@]}"; do
  label="${mode}_control64"
  checkpoint="$root/verl_checkpoints/three_directions_${mode}_control_64step/global_step_64/actor/huggingface"
  test -f "$checkpoint/config.json"
  bash tools/run_direction_eval.sh "$label" "$checkpoint" 1 2 \
    >"$run_dir/${label}.eval_launcher.log" 2>&1
  test -f "$eval_dir/$label.completed"
  labels+=("$label")
done

"$base/activevln_server_env/bin/python" tools/analyze_direction_eval.py \
  "${labels[@]}" --output matched_analysis.json \
  >"$run_dir/analysis.log" 2>&1
test -s "$eval_dir/matched_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
