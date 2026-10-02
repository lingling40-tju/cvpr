#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
training="$root/runlogs/fused_reward_group4_64step_seed11"
watcher="$root/runlogs/fused_followup_watcher"
mkdir -p "$watcher"
exec 9>"$watcher/watcher.lock"
flock -n 9 || { echo 'fused follow-up watcher already running' >&2; exit 2; }
if test -f "$watcher/watcher.completed"; then exit 0; fi
rm -f "$watcher/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$watcher/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$watcher/watcher.started"

until test -f "$training/completed"; do
  if test -f "$training/failed"; then
    echo '64-step fused reward training failed' >&2
    exit 1
  fi
  pid=$(cat "$root/runlogs/fused_group4_64_launcher.pid")
  kill -0 "$pid" 2>/dev/null || {
    echo "training launcher PID $pid stopped without completion marker" >&2
    exit 1
  }
  sleep 60
done

python3 "$root/tools/audit_fused_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 64 \
  >"$training/paired_train_audit.json"
test -s "$training/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$watcher/training_audited"

bash "$root/tools/run_fused_eval256.sh" >"$watcher/eval_launcher.log" 2>&1
test -f "$root/runlogs/fused_reward_group4_eval256/eval.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$watcher/watcher.completed"
