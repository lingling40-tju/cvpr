#!/usr/bin/env bash
set -euo pipefail

# Start the route-fidelity pilot only after the progress experiment concludes
# without a retained navigation gain. This watcher is idle while progress runs.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
progress_suite="$root/runlogs/three_direction_progress_scale_conditional"
progress_analysis="$root/runlogs/three_direction_full_val_unseen/scale_progress_128_analysis.json"
repeat_suite="$root/runlogs/three_direction_progress_full_replication"
repeat_analysis="$root/runlogs/three_direction_full_val_unseen_progress_replication_seed20261003/scale_progress_128_analysis.json"
run_dir="$root/runlogs/three_direction_route_fidelity_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'route fallback watcher already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/watcher.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

score_analysis() {
  "$base/activevln_server_env/bin/python" - "$1" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1]))
assert d['split']=='val_unseen' and d['episodes']==1839 and d['scenes']==11
assert d['mode']=='progress' and d['train_steps']==128
assert set(d['paired_seed_differences'])=={'11','22','33'}
assert set(d['models'])=={f'{arm}128_seed{seed}' for arm in ('progress','branch_control') for seed in (11,22,33)}
assert all(x['count']==1839 and x['inference_errors']==0 for x in d['models'].values())
sr,spl=d['mean_paired_sr_pp'],d['mean_paired_spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('positive_full_mean' if sr>0 and spl>=0 else 'nonpositive_full_mean')
PY
}

until test -f "$progress_suite/suite.completed"; do
  if test -f "$progress_suite/suite.failed"; then
    echo 'upstream progress suite failed' >&2; exit 1
  fi
  test -s "$progress_suite/suite.pid" || { echo 'missing progress suite PID' >&2; exit 1; }
  pid=$(cat "$progress_suite/suite.pid")
  kill -0 "$pid" 2>/dev/null || { echo "progress suite PID $pid stopped" >&2; exit 1; }
  sleep 60
done

if test -f "$progress_suite/branch_gain_retained"; then
  printf 'branch gain retained\n' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
  exit 0
fi
if test -f "$progress_suite/no_pilot_gain"; then
  decision=no_pilot_gain
else
  test -s "$progress_analysis"
  decision=$(score_analysis "$progress_analysis")
fi
printf '%s\n' "$decision" >"$run_dir/progress_decision.txt"
if [ "$decision" = positive_full_mean ]; then
  until test -f "$repeat_suite/suite.completed"; do
    if test -f "$repeat_suite/suite.failed"; then
      echo 'progress decode replication failed' >&2; exit 1
    fi
    test -s "$repeat_suite/suite.pid" || { echo 'missing replication PID' >&2; exit 1; }
    repeat_pid=$(cat "$repeat_suite/suite.pid")
    kill -0 "$repeat_pid" 2>/dev/null || { echo "replication PID $repeat_pid stopped" >&2; exit 1; }
    sleep 60
  done
  test ! -f "$repeat_suite/not_eligible"
  test -s "$repeat_analysis"
  second_decision=$(score_analysis "$repeat_analysis")
  printf '%s\n' "$second_decision" >"$run_dir/progress_replication_decision.txt"
  if [ "$second_decision" = positive_full_mean ]; then
    printf 'progress positive in both complete decodes\n' >"$run_dir/not_eligible"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
    exit 0
  fi
fi

bash tools/stage_route_fidelity_fallback.sh >"$run_dir/stage.log" 2>&1
bash tools/start_route_fidelity_service.sh 2 >"$run_dir/service.log" 2>&1
bash tools/run_route_fidelity_fallback.sh 64 0,1 11 >"$run_dir/train.log" 2>&1
bash tools/run_route_pilot_eval.sh >"$run_dir/eval.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
