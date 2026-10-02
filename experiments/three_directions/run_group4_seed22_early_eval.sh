#!/usr/bin/env bash
set -euo pipefail

# Use idle GPUs 0/1 to overlap completed seed-22 evaluation with seed-33
# group-size-four training on GPUs 2/3. The main suite sees completion markers.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/group4_seed22_early_eval"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'seed22 early eval already running' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f runlogs/group4_scale_conditional/train_seed22.completed
checkpoint="$root/verl_checkpoints/three_directions_group4_128step_seed22/global_step_128/actor/huggingface"
test -f "$checkpoint/config.json"
for count in 256 1839; do
  if test "$count" -eq 256; then
    result="$root/runlogs/three_direction_val256"
    expected=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
  else
    result="$root/runlogs/three_direction_full_val_unseen"
    expected=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$expected"
  test -f "$result/branch_control128_seed22.completed"
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT="$count" VLN_EVAL_PORT=8016 \
    bash tools/run_direction_eval.sh group4_128_seed22 "$checkpoint" 1 0 \
    >"$run/eval_${count}.log" 2>&1
  test -f "$result/group4_128_seed22.completed"
  "$base/activevln_server_env/bin/python" tools/analyze_matched_pair.py \
    --root "$result" --candidate group4_128_seed22 \
    --control branch_control128_seed22 --expected-count "$count" \
    --output "$result/paired_group4_128_seed22_vs_branch_control128_seed22.json" \
    >"$run/analyze_${count}.log" 2>&1
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval_${count}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
