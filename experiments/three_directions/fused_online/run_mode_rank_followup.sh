#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_mode_rank_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/mode_rank_followup"
train="$root/runlogs/mode_rank_group4_64step_seed11"
mkdir -p "$run"
exec 9>"$run/followup.lock"
flock -n 9 || { echo 'mode-rank followup already running' >&2; exit 2; }
if test -f "$run/followup.completed"; then exit 0; fi
rm -f "$run/followup.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/followup.failed"; fi
}
trap on_exit EXIT
test -s "$root/runlogs/mode_rank_64step.launcher.pid"
trainer_pid=$(cat "$root/runlogs/mode_rank_64step.launcher.pid")
while ! test -f "$train/completed"; do
  if test -f "$train/failed"; then echo 'mode-rank training failed' >&2; exit 1; fi
  if ! kill -0 "$trainer_pid" 2>/dev/null; then
    echo "mode-rank launcher $trainer_pid vanished without completion" >&2; exit 1
  fi
  sleep 20
done
if ! test -f "$run/training_audited"; then
  python "$root/tools/audit_mode_rank_pilot.py" \
    --root "$root" --source-root "$source_root" --steps 64 \
    --output "$train/paired_train_audit.json" >"$run/train_audit.log" 2>&1
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training_audited"
fi
bash "$root/tools/run_mode_rank_eval256.sh" >"$run/eval_launcher.log" 2>&1
test -f "$root/runlogs/mode_rank_eval256/eval.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/followup.completed"
