#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/stopaware_followup"
mkdir -p "$run"
exec 9>"$run/followup.lock"
flock -n 9 || { echo 'stop-aware follow-up already active' >&2; exit 2; }
if test -f "$run/followup.completed"; then exit 0; fi
rm -f "$run/followup.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/followup.failed"; fi
}
trap on_exit EXIT
smoke="$root/runlogs/stopaware_group4_2step_seed11"
until test -f "$smoke/completed"; do
  if test -f "$smoke/failed"; then echo 'stop-aware smoke failed' >&2; exit 1; fi
  pid=$(cat "$run/smoke.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'stop-aware smoke launcher stopped' >&2; exit 1; }
  sleep 20
done
python3 "$root/tools/audit_stopaware_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 2 \
  >"$smoke/paired_train_audit.json"
test -s "$smoke/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/smoke_audited"

bash "$root/tools/run_stopaware_pilot.sh" 64 >"$run/train64.launcher.log" 2>&1
training="$root/runlogs/stopaware_group4_64step_seed11"
test -f "$training/completed"
python3 "$root/tools/audit_stopaware_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 64 \
  >"$training/paired_train_audit.json"
test -s "$training/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training_audited"

bash "$root/tools/run_stopaware_eval256.sh" >"$run/eval256.launcher.log" 2>&1
test -f "$root/runlogs/stopaware_eval256/eval.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/followup.completed"
