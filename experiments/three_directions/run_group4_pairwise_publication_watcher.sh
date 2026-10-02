#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_group4_pairwise_20261002"
eval_suite="$root/runlogs/group4_pairwise_scale_eval_conditional"
gate="$root/runlogs/group4_pairwise_scale_conditional"
run_dir="$root/runlogs/group4_pairwise_publication_watcher"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'pairwise publication watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

until test -f "$eval_suite/suite.completed"; do
  if test -f "$eval_suite/suite.failed"; then echo 'pairwise full evaluation failed' >&2; exit 1; fi
  pid_file="$root/runlogs/group4_pairwise_scale_eval_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "evaluation PID $pid stopped" >&2; exit 1; }
  sleep 60
done

decision=$(cat "$gate/pilot_decision.txt")
if [ "$decision" = ineligible ]; then
  test -f "$gate/no_pilot_gain"
  test -f "$eval_suite/no_eligible_pilot"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_eligible_pilot"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
  exit 0
fi
test "$decision" = eligible
bash "$root/tools/package_group4_pairwise_scaled.sh" \
  >"$run_dir/package.log" 2>&1
test -f "$root/runlogs/group4_pairwise_scale_publication/package.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
