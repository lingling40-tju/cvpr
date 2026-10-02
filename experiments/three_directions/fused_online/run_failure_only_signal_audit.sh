#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_fused_reward_20261003
training="$root/runlogs/failure_only_group4_64step_seed11"
run="$root/runlogs/failure_only_signal_audit"
mkdir -p "$run"
exec 9>"$run/audit.lock"
flock -n 9 || { echo 'failure-only signal audit already active' >&2; exit 2; }
if test -f "$run/audit.completed"; then exit 0; fi
rm -f "$run/audit.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/audit.failed"; fi
}
trap on_exit EXIT
until test -f "$training/completed"; do
  if test -f "$training/failed"; then echo '64-step training failed' >&2; exit 1; fi
  if test -f "$root/runlogs/failure_only_followup/followup.failed"; then
    echo 'failure-only follow-up failed' >&2; exit 1
  fi
  pid=$(cat "$root/runlogs/failure_only_followup/launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'training watcher stopped' >&2; exit 1; }
  sleep 60
done
python3 "$root/tools/analyze_failure_only_onpolicy.py" \
  --rollout "$root/verl_checkpoints/failure_only_group4_64step_seed11/rollout.jsonl" \
  --expected-steps 64 \
  --output "$run/onpolicy_signal64.json" >"$run/analyze.log" 2>&1
python3 - "$run/onpolicy_signal64.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='failure_only_group4_training_signal_v1'
assert x['steps']==64 and x['group_size']==4 and x['episode_groups']==256
assert x['successful_rollouts']+x['unsuccessful_rollouts']==1024
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/audit.completed"
