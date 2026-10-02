#!/usr/bin/env bash
set -euo pipefail

# Reuse GPU 0/1 after the compute-matched pilot releases them. The regular
# scale evaluator uses GPU 2/3 and skips these labels after they complete.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
pairwise="$base/ActiveVLN_group4_pairwise_20261002/runlogs/group4_pairwise_64pilot"
run_dir="$root/runlogs/group4_early_eval_after_pairwise"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'early group4 evaluator already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"

until test -f "$pairwise/watcher.completed"; do
  if test -f "$pairwise/watcher.failed"; then
    echo 'pairwise pilot failed before releasing GPU 0/1' >&2
    exit 1
  fi
  pid_file="$base/ActiveVLN_group4_pairwise_20261002/runlogs/group4_pairwise_64pilot.launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "pairwise pilot watcher PID $pid stopped without completion" >&2
    exit 1
  fi
  sleep 60
done
test -f "$pairwise/service.stopped"
test -f "$pairwise/eval.completed"

experiment=three_directions_group4_128step
checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
test -f "$root/runlogs/$experiment/completed"
test -s "$root/runlogs/$experiment/paired_train_audit.json"
test -f "$checkpoint/config.json"
cd "$root"
label=group4_128_seed11
control=branch_control128_seed11
for count in 256 1839; do
  if [ "$count" = 256 ]; then
    result="$root/runlogs/three_direction_val256"
    expected_sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
  else
    result="$root/runlogs/three_direction_full_val_unseen"
    expected_sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$expected_sha"
  test -f "$result/$control.completed"
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT="$count" VLN_EVAL_PORT=8019 \
    bash tools/run_direction_eval.sh "$label" "$checkpoint" 1 0 \
    >"$run_dir/eval_${count}.launcher.log" 2>&1
  test -f "$result/$label.completed"
  pair="$result/paired_${label}_vs_${control}.json"
  "$base/activevln_server_env/bin/python" tools/analyze_matched_pair.py \
    --root "$result" --candidate "$label" --control "$control" \
    --expected-count "$count" --output "$pair" \
    >"$run_dir/analyze_${count}.log" 2>&1
  test -s "$pair"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/eval_${count}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
