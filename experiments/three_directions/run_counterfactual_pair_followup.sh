#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
finalizer="$root/runlogs/three_direction_pilot_recovery"
matched="$root/runlogs/three_direction_matched_followup"
target="$root/runlogs/three_direction_val_counterfactual"
exec 9>"$target/suite.lock"
flock -n 9 || { echo 'counterfactual pair evaluation already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$target/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$target/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$target/suite.started"

until test -f "$finalizer/finalizer.completed"; do
  if test -f "$finalizer/finalizer.failed"; then
    echo 'pilot recovery finalizer failed' >&2
    exit 1
  fi
  pid=$(cat "$finalizer/finalizer.pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "pilot recovery finalizer PID $pid stopped" >&2
    exit 1
  fi
  sleep 60
done
until test -f "$matched/suite.completed"; do
  if test -f "$matched/suite.failed"; then
    echo 'matched-control follow-up failed' >&2
    exit 1
  fi
  pid=$(cat "$matched/suite.pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "matched-control PID $pid stopped" >&2
    exit 1
  fi
  sleep 60
done

model="$root/verl_checkpoints/three_directions_counterfactual_64step/global_step_64/actor/huggingface"
test -f "$root/runlogs/three_directions_counterfactual_64step/completed"
test -f "$model/config.json"
test -f "$target/manifest.json"
VLN_EVAL_RESULT_ROOT="$target" VLN_EVAL_COUNT=144 \
  bash tools/run_direction_eval.sh counterfactual64_pairs "$model" 1 2 \
  >"$target/counterfactual64_pairs.launcher.log" 2>&1
test -f "$target/counterfactual64_pairs.completed"
"/Knowin/foundation/haozhiwang/whz/activevln_server_env/bin/python" \
  tools/analyze_counterfactual_val_pairs.py --candidate counterfactual64_pairs \
  >"$target/analysis.log" 2>&1
test -s "$target/analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$target/suite.completed"
