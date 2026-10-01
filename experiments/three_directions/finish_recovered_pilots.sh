#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
train_dir="$root/runlogs/three_direction_pilot_suite"
early_dir="$root/runlogs/three_direction_early_eval"
eval_dir="$root/runlogs/three_direction_val256"
matched_dir="$root/runlogs/three_direction_matched_followup"
recovery_dir="$root/runlogs/three_direction_pilot_recovery"
exec 9>"$recovery_dir/finalizer.lock"
flock -n 9 || { echo 'recovery finalizer already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$recovery_dir/finalizer.failed"; fi
}
trap on_exit EXIT
rm -f "$recovery_dir/finalizer.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$recovery_dir/finalizer.started"

until test -f "$root/runlogs/three_directions_counterfactual_64step/completed"; do
  if test -f "$root/runlogs/three_directions_counterfactual_64step/failed"; then
    echo 'counterfactual pilot failed; inspect it before resuming evaluation' >&2
    exit 1
  fi
  cf_pid=$(cat "$recovery_dir/counterfactual.pid")
  if ! kill -0 "$cf_pid" 2>/dev/null; then
    echo "counterfactual PID $cf_pid stopped without completion" >&2
    exit 1
  fi
  sleep 60
done
test -s "$root/runlogs/three_directions_counterfactual_64step/validation.json"
test -f "$root/verl_checkpoints/three_directions_counterfactual_64step/global_step_64/actor/huggingface/config.json"

until test -f "$early_dir/suite.completed"; do
  if test -f "$early_dir/suite.failed"; then
    echo 'early held-out evaluation failed; inspect before restarting shared port' >&2
    exit 1
  fi
  early_pid=$(cat "$early_dir/suite.pid")
  if ! kill -0 "$early_pid" 2>/dev/null; then
    echo "early evaluation PID $early_pid stopped without completion" >&2
    exit 1
  fi
  sleep 60
done

for mode in branch recovery; do
  run="$root/runlogs/three_directions_${mode}_64step"
  test -s "$run/checkpoint_validated.json"
  test "$(cat "$run/failed")" = 127
  mv "$run/failed" "$run/failed.wrapper_parse_127"
  cp "$run/checkpoint_validated.json" "$run/validation.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
done
test -f "$train_dir/suite.failed"
mv "$train_dir/suite.failed" "$train_dir/suite.failed.wrapper_parse_127"
"/Knowin/foundation/haozhiwang/whz/activevln_train_env/bin/python" - <<'PY'
import json
from pathlib import Path
root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
result = {
    'cause': 'The running shell wrappers reread run_direction_pilot.sh while the file was overwritten in place; they exited 127 after saving their step-64 checkpoints.',
    'accepted_checkpoint_evidence': [
        'branch: 64 contiguous rollout steps, 64 TensorBoard scalar steps, nonzero actor gradients, complete model shards',
        'recovery: same four checks',
    ],
    'counterfactual': 'Completed by a fresh wrapper using the corrected frozen script',
    'stdout_limitation': 'The branch and recovery train.log files were truncated to the shell parse error. Original TensorBoard and rollout artifacts were retained.',
    'original_failure_markers': 'renamed to failed.wrapper_parse_127; not erased',
}
p = root / 'runlogs/three_direction_pilot_suite/suite.recovery.json'
p.write_text(json.dumps(result, indent=2) + '\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$train_dir/suite.completed"

if curl -fsS --max-time 2 http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
  echo 'evaluation port 8010 still in use after early evaluation' >&2
  exit 1
fi
nohup bash tools/run_direction_eval_suite.sh >"$eval_dir/suite.restarted.log" 2>&1 </dev/null &
eval_pid=$!
echo "$eval_pid" >"$eval_dir/suite.pid"
sleep 3
kill -0 "$eval_pid"
test ! -f "$eval_dir/suite.failed"

nohup bash tools/run_matched_control_followup.sh >"$matched_dir/suite.restarted.log" 2>&1 </dev/null &
matched_pid=$!
echo "$matched_pid" >"$matched_dir/suite.pid"
sleep 3
kill -0 "$matched_pid"
test ! -f "$matched_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$recovery_dir/finalizer.completed"
