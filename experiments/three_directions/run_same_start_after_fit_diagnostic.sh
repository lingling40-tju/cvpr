#!/usr/bin/env bash
set -euo pipefail

# Use GPU 0 after the predecessor and its fit/development diagnosis
# release it. The n=4 navigation trainer retains GPUs 2/3.
base=/Knowin/foundation/haozhiwang/whz
prior="$base/policy_unbounded_expert_cross_goal_potential_lora_20261004"
scratch="$base/policy_same_start_relative_lora_20261004"
scale="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_scale"
run="$scratch/runlogs/after_fit_diagnostic"
diagnosis="$prior/runlogs/fit_diagnostic"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'same-start relative watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
test -f "$scratch/runlogs/preflight/completed"
while ! test -f "$diagnosis/completed"; do
  test ! -f "$diagnosis/failed" || {
    echo 'fit/development diagnosis failed; repair before new fit' >&2; exit 1;
  }
  test -s "$diagnosis/launcher.pid"
  pid=$(cat "$diagnosis/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_unbounded_expert_fit_diagnostic.sh' >/dev/null || {
      echo 'fit/development diagnosis watcher vanished' >&2; exit 1;
    }
  sleep 30
done
if test -f "$diagnosis/skipped_after_pass"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_after_prior_pass"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
test -f "$prior/runlogs/full/failed_development_gate"
test -s "$diagnosis/fit_vs_development.json"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n '1p' | tr -d ' ')
  test -n "$used"
  if test "$used" -lt 30000; then break; fi
  sleep 30
done
if ! test -f "$scale/training.completed"; then
  curl -fsS --max-time 3 http://127.0.0.1:5013/health \
    >"$run/habitat_health_before.json"
fi
VLN_REP_GPU=0 VLN_REP_MAX_USED_MIB=30000 \
  bash "$scratch/run_same_start_relative_lora.sh" smoke \
  >"$run/smoke.launcher.log" 2>&1
test -f "$scratch/runlogs/smoke/completed"
VLN_REP_GPU=0 VLN_REP_MAX_USED_MIB=30000 \
  bash "$scratch/run_same_start_relative_lora.sh" full \
  >"$run/full.launcher.log" 2>&1
test -f "$scratch/runlogs/full/completed"
test -s "$scratch/runlogs/full/development.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
