#!/usr/bin/env bash
set -euo pipefail

# Wait for every audited training/evaluation stage before creating compact
# paper inputs. This watcher never launches inference or changes checkpoints.
root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
training="$root/runlogs/three_direction_scale_branch_128step_suite"
audits="$root/runlogs/three_direction_scale_branch_128step_pair_audit"
screen="$root/runlogs/three_direction_scale_branch_128step_eval"
full="$root/runlogs/three_direction_scale_branch_128step_full_eval_all"
output="$root/runlogs/three_direction_scale_branch_128step_publication"
run_dir="$root/runlogs/three_direction_scale_branch_128step_publication_watcher"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'publication watcher already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/watcher.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

while :; do
  ready=1
  for spec in "$training:suite:suite" "$audits:watcher:watcher" \
              "$screen:suite:suite" "$full:suite:suite"; do
    folder=${spec%%:*}
    tail=${spec#*:}
    label=${tail%%:*}
    pid_label=${tail#*:}
    if test -f "$folder/$label.failed"; then
      echo "upstream failed: $folder/$label.failed" >&2
      exit 1
    fi
    if ! test -f "$folder/$label.completed"; then
      ready=0
      pid=$(cat "$folder/$pid_label.pid")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "upstream PID $pid stopped: $folder" >&2
        exit 1
      fi
    fi
  done
  if [ "$ready" -eq 1 ]; then break; fi
  sleep 60
done

bash tools/package_scaled_results.sh >"$run_dir/package.log" 2>&1
test -f "$output/package.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
