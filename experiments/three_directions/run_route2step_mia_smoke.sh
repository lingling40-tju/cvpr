#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
model_root="$base/route2step_mia_20261004"
scale="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_scale"
run="$model_root/runlogs/smoke"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'Route2Step MIA smoke already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$model_root/runlogs/download/completed"
test ! -f "$scale/control_seed33.completed"
used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
  sed -n '2p' | tr -d ' ')
test -n "$used1" && test "$used1" -lt 8000
! curl -fsS --max-time 1 http://127.0.0.1:8122/health >/dev/null 2>&1
sha256sum "$model_root/smoke_route2step_mia_progress.py" \
  "$root/runlogs/ordinal_progress/policy_process_manifest.json" \
  "$model_root/model/MIA/config.json" >"$run/source.sha256"
CUDA_VISIBLE_DEVICES=1 "$base/activevln_train_env/bin/python" \
  "$model_root/smoke_route2step_mia_progress.py" \
  --model "$model_root/model/MIA" \
  --manifest "$root/runlogs/ordinal_progress/policy_process_manifest.json" \
  --record-root "$root/runlogs/ordinal_progress/policy_process_turns" \
  --output "$run/format.json" >"$run/smoke.log" 2>&1
test -s "$run/format.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
