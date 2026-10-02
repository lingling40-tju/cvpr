#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/failure_only_followup"
eval_run="$root/runlogs/failure_only_group4_eval256"
result="$source_root/runlogs/three_direction_val256"
mkdir -p "$run" "$eval_run"
exec 9>"$run/followup.lock"
flock -n 9 || { echo 'failure-only follow-up already active' >&2; exit 2; }
if test -f "$run/followup.completed"; then exit 0; fi
rm -f "$run/followup.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/followup.failed"; fi
}
trap on_exit EXIT
test -f "$root/runlogs/fused_scale_conditional/no_pilot_gain"
test -f "$root/runlogs/fused_scale_conditional/suite.completed"

smoke="$root/runlogs/failure_only_group4_2step_seed11"
until test -f "$smoke/completed"; do
  if test -f "$smoke/failed"; then echo 'failure-only smoke failed' >&2; exit 1; fi
  pid=$(cat "$root/runlogs/failure_only_2step_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'smoke launcher stopped' >&2; exit 1; }
  sleep 30
done
python3 "$root/tools/audit_failure_only_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 2 \
  >"$smoke/paired_train_audit.json"
test -s "$smoke/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/smoke_audited"

bash "$root/tools/run_failure_only_pilot.sh" 64 \
  >"$run/train64.launcher.log" 2>&1
training="$root/runlogs/failure_only_group4_64step_seed11"
test -f "$training/completed"
python3 "$root/tools/audit_failure_only_pilot.py" \
  --root "$root" --source-root "$source_root" --steps 64 \
  >"$training/paired_train_audit.json"
test -s "$training/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training_audited"

test -f "$result/group4_64_seed11.completed"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
label=failure_only_group4_64_seed11
checkpoint="$root/verl_checkpoints/failure_only_group4_64step_seed11/global_step_64/actor/huggingface"
test -f "$checkpoint/config.json"
VLN_EVAL_PORT=8014 bash "$source_root/tools/run_direction_eval.sh" \
  "$label" "$checkpoint" 3 2 >"$eval_run/eval.log" 2>&1
test -f "$result/$label.completed"
"$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$label" --control group4_64_seed11 \
  --expected-count 256 \
  --output "$eval_run/paired_failure_only_vs_group4.json" \
  >"$eval_run/analysis.log" 2>&1
test -s "$eval_run/paired_failure_only_vs_group4.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$eval_run/eval.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/followup.completed"
