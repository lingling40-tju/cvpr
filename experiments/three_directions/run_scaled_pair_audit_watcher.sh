#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
suite="$root/runlogs/three_direction_scale_branch_128step_suite"
run_dir="$root/runlogs/three_direction_scale_branch_128step_pair_audit"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'pair audit watcher already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/watcher.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

for seed in 11 22 33; do
  until test -f "$suite/seed${seed}.completed"; do
    if test -f "$suite/suite.failed"; then echo 'scaled training failed' >&2; exit 1; fi
    suite_pid=$(cat "$suite/suite.pid")
    if ! kill -0 "$suite_pid" 2>/dev/null; then
      echo "scaled training suite PID $suite_pid stopped before seed $seed" >&2
      exit 1
    fi
    sleep 60
  done
  "$base/activevln_train_env/bin/python" tools/audit_scaled_training_pair.py \
    --root . --dataset data/branch_scale512_train.parquet \
    --seed "$seed" --steps 128 \
    --output "$suite/seed${seed}_pair_audit.json" \
    >"$run_dir/seed${seed}.log" 2>&1
  test -s "$suite/seed${seed}_pair_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/seed${seed}.audited"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
