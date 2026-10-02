#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
eval_suite="$root/runlogs/optimizer_scale_eval_conditional"
run_dir="$root/runlogs/optimizer_publication_watcher"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'optimizer publication watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

until test -f "$eval_suite/suite.completed"; do
  if test -f "$eval_suite/suite.failed"; then
    echo 'optimizer evaluation failed' >&2
    exit 1
  fi
  pid_file="$root/runlogs/optimizer_scale_eval_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "evaluation watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done

for mode in group4 kl_anchor; do
  gate="$root/runlogs/${mode}_scale_conditional"
  decision=$(cat "$gate/pilot_decision.txt")
  if [ "$decision" = eligible ]; then
    bash "$root/tools/package_optimizer_scaled.sh" "$mode" \
      >"$run_dir/package_${mode}.log" 2>&1
    test -f "$root/runlogs/${mode}_scale_publication/package.completed"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/package_${mode}.completed"
  else
    test "$decision" = ineligible
    test -f "$gate/no_pilot_gain"
  fi
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
