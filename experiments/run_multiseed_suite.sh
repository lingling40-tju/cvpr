#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930
run_dir="$root/runlogs/eventtrace_r2r64"
mkdir -p "$run_dir"
for seed in 11 22 33; do
  for arm in control event; do
    label="seed${seed}_${arm}"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.started"
    if bash "$root/tools/run_multiseed_train.sh" "$seed" "$arm" \
      >"$run_dir/$label.log" 2>&1; then
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.completed"
    else
      status=$?
      printf '%s\n' "$status" >"$run_dir/$label.failed"
      exit "$status"
    fi
  done
done
