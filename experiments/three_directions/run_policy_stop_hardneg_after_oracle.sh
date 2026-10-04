#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
scratch="$base/policy_stop_hardneg_20261004"
oracle="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_pilot"
run="$scratch/runlogs/after_oracle"
full="$scratch/runlogs/full"
mkdir -p "$run"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'hard-negative STOP watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$oracle/completed"; do
  if test -f "$oracle/failed"; then
    echo 'oracle pilot failed; do not launch competing model fit' >&2
    exit 1
  fi
  if test -s "$oracle/launcher.pid" && \
      ! kill -0 "$(cat "$oracle/launcher.pid")" 2>/dev/null; then
    echo 'oracle launcher disappeared without terminal marker' >&2
    exit 1
  fi
  sleep 30
done

# The oracle's completed marker follows both val-unseen evaluation lanes.
# STOP-pair's patched watcher waits for GPU 1 again before its own eval.
test "$(sha256sum "$base/ActiveVLN_stop_pair_group4_20261004/tools/run_stop_pair_after_oracle.sh" | awk '{print $1}')" = \
  d93c9b0607f9a3ad728096ec54ece34ec1d2e4fcf28f7d61b7b4c2f38574b318
while true; do
  if test -f "$full/completed"; then
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
    exit 0
  fi
  used1=$(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
  test -n "$used1"
  if test "$used1" -lt 8000 && \
      ! curl -fsS --max-time 2 http://127.0.0.1:8092/v1/models >/dev/null 2>&1 && \
      ! curl -fsS --max-time 2 http://127.0.0.1:8094/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 15
done

bash "$scratch/run_policy_stop_hardneg_lora.sh" full \
  >"$run/full_launcher.log" 2>&1
test -f "$full/completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
