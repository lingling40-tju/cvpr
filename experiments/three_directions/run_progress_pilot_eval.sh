#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
progress_root="$base/ActiveVLN_progress_fallback_20261002"
full_eval="$source_root/runlogs/three_direction_scale_branch_128step_full_eval_all"
result="$source_root/runlogs/three_direction_val256"
run_dir="$progress_root/runlogs/progress_pilot_eval"
label=progress64_seed11
checkpoint="$progress_root/verl_checkpoints/three_directions_progress_fallback_64step/global_step_64/actor/huggingface"
analysis="$result/paired_progress64_vs_branch_control64.json"

test -f "$full_eval/suite.completed" || { echo 'complete branch evaluation still running' >&2; exit 2; }
test -f "$progress_root/runlogs/three_directions_progress_fallback_64step/completed"
test -s "$progress_root/runlogs/three_directions_progress_fallback_64step/validation.json"
test -f "$checkpoint/config.json"
test -f "$result/branch_control64.completed"
actual_manifest_sha=$(sha256sum "$result/manifest.json" | awk '{print $1}')
[ "$actual_manifest_sha" = 546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46 ] || {
  echo 'fixed 256-episode manifest hash mismatch' >&2; exit 1;
}

mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'progress pilot evaluation already active' >&2; exit 2; }
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

VLN_EVAL_PORT=8013 bash "$source_root/tools/run_direction_eval.sh" \
  "$label" "$checkpoint" 1 2 >"$run_dir/$label.launcher.log" 2>&1
test -f "$result/$label.completed"
"$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$label" --control branch_control64 \
  --expected-count 256 --output "$analysis" >"$run_dir/analysis.log" 2>&1
test -s "$analysis"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
