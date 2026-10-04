#!/usr/bin/env bash
set -euo pipefail

# Use the same GPU-0 representation lane only if the simpler ablation
# misses its unchanged development gate. Keep the n=4 RL experiment
# and frozen val-unseen evaluation independent of this fit.
base=/Knowin/foundation/haozhiwang/whz
prior="$base/policy_unbounded_cross_goal_potential_lora_20261004/runlogs/full"
root="$base/policy_unbounded_expert_cross_goal_potential_lora_20261004"
run="$root/runlogs/after_unbounded"
scale="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_scale"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'expert potential watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$prior/completed"; do
  test ! -f "$prior/failed" || {
    echo 'unbounded predecessor fit failed mechanically' >&2; exit 1;
  }
  test -s "$prior/launcher.pid"
  pid=$(cat "$prior/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_unbounded_cross_goal_potential_lora.sh full' >/dev/null || {
      echo 'unbounded predecessor launcher vanished' >&2; exit 1;
    }
  sleep 30
done
if test -f "$prior/passed_development_gate"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_after_prior_pass"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
test -f "$prior/failed_development_gate"
if test -s "$prior/launcher.pid"; then
  pid=$(cat "$prior/launcher.pid")
  while ps -o args= -p "$pid" | grep -F \
    'run_unbounded_cross_goal_potential_lora.sh full' >/dev/null; do
    sleep 10
  done
fi

# The predecessor used GPU 0 alongside the live Habitat service and
# four validation shards. Preserve the same tested headroom.
while true; do
  used=$(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
  test -n "$used"
  if test "$used" -lt 30000; then break; fi
  sleep 30
done
if ! test -f "$scale/training.completed"; then
  curl -fsS --max-time 3 http://127.0.0.1:5013/health \
    >"$run/habitat_health_before.json"
fi
test -f "$root/runlogs/preflight/completed"
test -f "$root/runlogs/smoke/completed"
VLN_REP_GPU=0 VLN_REP_MAX_USED_MIB=30000 \
  bash "$root/run_unbounded_expert_cross_goal_potential_lora.sh" full \
  >"$run/full.launcher.log" 2>&1
test -f "$root/runlogs/full/completed"
test -s "$root/runlogs/full/development.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
