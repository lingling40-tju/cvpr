#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
result="$root/runlogs/three_direction_val256"
run_dir="$root/runlogs/kl_anchor_pilot_eval"
training="$root/runlogs/three_directions_kl_anchor_64step_seed11"
checkpoint="$root/verl_checkpoints/three_directions_kl_anchor_64step_seed11/global_step_64/actor/huggingface"
label=kl_anchor_64_seed11
analysis="$result/paired_kl_anchor_64_vs_branch_control64.json"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'KL pilot evaluation already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT

test -f "$training/completed"
test -s "$training/paired_train_audit.json"
test -f "$checkpoint/config.json"
test -f "$result/branch_control64.completed"
manifest_sha=$(sha256sum "$result/manifest.json" | awk '{print $1}')
[ "$manifest_sha" = 546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46 ]
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

VLN_EVAL_PORT=8015 bash "$root/tools/run_direction_eval.sh" \
  "$label" "$checkpoint" 1 0 >"$run_dir/$label.launcher.log" 2>&1
test -f "$result/$label.completed"
"$base/activevln_server_env/bin/python" "$root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$label" --control branch_control64 \
  --expected-count 256 --output "$analysis" >"$run_dir/control_analysis.log" 2>&1
test -s "$analysis"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
