#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
training="$root/runlogs/three_directions_kl_anchor_64step_seed11"
run_dir="$root/runlogs/kl_anchor_eval_watcher"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'KL eval watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

until test -f "$training/completed"; do
  if test -f "$training/failed"; then
    echo 'KL training failed' >&2
    exit 1
  fi
  test -s "$root/runlogs/kl_anchor_64_launcher.pid" || { echo 'missing trainer PID' >&2; exit 1; }
  pid=$(cat "$root/runlogs/kl_anchor_64_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "trainer PID $pid stopped" >&2; exit 1; }
  sleep 60
done

bash "$root/tools/run_kl_anchor_pilot_eval.sh" >"$run_dir/eval.log" 2>&1
test -f "$root/runlogs/kl_anchor_pilot_eval/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
