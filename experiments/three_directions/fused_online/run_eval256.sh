#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/fused_reward_group4_eval256"
result="$source_root/runlogs/three_direction_val256"
label=fused_group4_64_seed11
checkpoint="$root/verl_checkpoints/fused_reward_group4_64step_seed11/global_step_64/actor/huggingface"
mkdir -p "$run"
exec 9>"$run/eval.lock"
flock -n 9 || { echo 'fused 256 evaluation already running' >&2; exit 2; }
if test -f "$run/eval.completed"; then exit 0; fi
rm -f "$run/eval.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/eval.failed"; fi
}
trap on_exit EXIT
test -f "$root/runlogs/fused_reward_group4_64step_seed11/completed"
test -s "$root/runlogs/fused_reward_group4_64step_seed11/paired_train_audit.json"
test -f "$result/group4_64_seed11.completed"
test -f "$checkpoint/config.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46

VLN_EVAL_PORT=8014 bash "$source_root/tools/run_direction_eval.sh" \
  "$label" "$checkpoint" 3 2 >"$run/eval.log" 2>&1
test -f "$result/$label.completed"
"$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$label" --control group4_64_seed11 \
  --expected-count 256 --output "$run/paired_fused_vs_group4.json" \
  >"$run/analysis.log" 2>&1
test -s "$run/paired_fused_vs_group4.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval.completed"
