#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
scale="$root/runlogs/dynamic_scale_conditional"
run_dir="$root/runlogs/dynamic_publication_watcher"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'dynamic publication watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

until test -f "$scale/suite.completed"; do
  if test -f "$scale/suite.failed"; then echo 'dynamic scale failed' >&2; exit 1; fi
  pid_file="$root/runlogs/dynamic_scale_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "dynamic scale PID $pid stopped" >&2; exit 1; }
  sleep 60
done
decision=$(cat "$scale/pilot_decision.txt")
if [ "$decision" = eligible ]; then
  bash "$root/tools/package_optimizer_scaled.sh" dynamic \
    >"$run_dir/package.log" 2>&1
  test -f "$root/runlogs/dynamic_scale_publication/package.completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/package.completed"
else
  test "$decision" = ineligible
  test -f "$scale/no_pilot_gain"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
