#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/signed_timeout_probe"
mkdir -p "$run"
exec 9>"$run/probe.lock"
flock -n 9 || { echo 'signed timeout probe already active' >&2; exit 2; }
if test -f "$run/probe.completed"; then exit 0; fi
rm -f "$run/probe.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/probe.failed"; fi
}
trap on_exit EXIT
test -f "$root/runlogs/failure_only_group4_64step_seed11/completed"
"$base/activevln_train_env/bin/python" \
  "$root/tools/probe_signed_timeout_reward.py" \
  --rollout "$root/verl_checkpoints/failure_only_group4_64step_seed11/rollout.jsonl" \
  --output "$run/development.json" >"$run/probe.log" 2>&1
test -s "$run/development.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/probe.completed"
