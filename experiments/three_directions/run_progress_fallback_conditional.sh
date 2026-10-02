#!/usr/bin/env bash
set -euo pipefail

# Continue to the isolated progress-reward pilot only if the complete branch
# comparison (and, when eligible, its second decode pass) fails the
# predeclared paired SR/SPL sign rule. Partial episode counts never trigger it.
root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
full_suite="$root/runlogs/three_direction_scale_branch_128step_full_eval_all"
full_result="$root/runlogs/three_direction_full_val_unseen/scale_branch_128_analysis.json"
repeat_suite="$root/runlogs/three_direction_scale_branch_128step_full_replication"
repeat_result="$root/runlogs/three_direction_full_val_unseen_replication_seed20261003/scale_branch_128_analysis.json"
run_dir="$root/runlogs/three_direction_progress_fallback_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'progress fallback watcher already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/watcher.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

wait_for_suite() {
  local folder=$1 pid
  until test -f "$folder/suite.completed"; do
    if test -f "$folder/suite.failed"; then
      echo "upstream suite failed: $folder" >&2
      exit 1
    fi
    test -s "$folder/suite.pid" || { echo "missing upstream PID: $folder" >&2; exit 1; }
    pid=$(cat "$folder/suite.pid")
    kill -0 "$pid" 2>/dev/null || { echo "upstream PID $pid stopped: $folder" >&2; exit 1; }
    sleep 60
  done
}

decision() {
  "$base_python" - "$1" <<'PY'
import json
import math
import sys

data = json.load(open(sys.argv[1]))
assert data['split'] == 'val_unseen' and data['episodes'] == 1839
assert data['mode'] == 'branch' and data['train_steps'] == 128
assert data['scenes'] == 11
expected = {f'{arm}128_seed{seed}' for arm in ('branch', 'branch_control')
            for seed in (11, 22, 33)}
assert set(data['models']) == expected
assert set(data['paired_seed_differences']) == {'11', '22', '33'}
assert all(model['count'] == 1839 and model['inference_errors'] == 0
           for model in data['models'].values())
sr = data['mean_paired_sr_pp']
spl = data['mean_paired_spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('positive' if sr > 0 and spl >= 0 else 'nonpositive')
PY
}

base_python=/Knowin/foundation/haozhiwang/whz/activevln_server_env/bin/python
wait_for_suite "$full_suite"
test -s "$full_result"
first=$(decision "$full_result")
printf '%s\n' "$first" >"$run_dir/first_full_decision.txt"

if [ "$first" = positive ]; then
  wait_for_suite "$repeat_suite"
  if test -f "$repeat_suite/not_eligible"; then
    echo 'first pass positive but second pass marked not eligible' >&2
    exit 1
  fi
  test -s "$repeat_result"
  second=$(decision "$repeat_result")
  printf '%s\n' "$second" >"$run_dir/second_full_decision.txt"
  if [ "$second" = positive ]; then
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/branch_gain_retained"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
    exit 0
  fi
fi

# The full comparison has ruled out a retained branch gain. The stage script
# checks source and patch hashes and refuses to alter the original project.
bash tools/stage_progress_fallback.sh >"$run_dir/stage.log" 2>&1
bash tools/start_progress_service.sh 3 >"$run_dir/service.log" 2>&1
bash tools/run_progress_fallback.sh 64 0,1 11 >"$run_dir/train.log" 2>&1
bash tools/run_progress_pilot_eval.sh >"$run_dir/eval.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
