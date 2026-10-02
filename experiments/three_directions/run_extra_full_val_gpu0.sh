#!/usr/bin/env bash
set -euo pipefail

# Third model lane for the two seed-33 full evaluations. The existing suite
# acquires the same per-label lock and skips a completed label when it reaches
# seed 33, so a model is never evaluated twice concurrently.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
result="$root/runlogs/three_direction_full_val_unseen"
main_suite="$root/runlogs/three_direction_scale_branch_128step_full_eval_all"
run_dir="$root/runlogs/three_direction_scale_branch_128step_full_eval_extra_gpu0"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'extra full-evaluation lane already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
test -f "$main_suite/suite.started"
test -f "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
if curl -fsS --max-time 1 http://127.0.0.1:8013/v1/models >/dev/null 2>&1; then
  echo 'port 8013 already in use' >&2
  exit 1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

for arm in branch_control branch; do
  label="${arm}128_seed33"
  experiment="three_directions_${arm}_128step_seed33"
  checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
  test -f "$root/runlogs/$experiment/completed"
  test -f "$checkpoint/config.json"
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT=1839 VLN_EVAL_PORT=8013 \
    bash tools/run_direction_eval.sh "$label" "$checkpoint" 0 2 \
    >"$run_dir/$label.launcher.log" 2>&1
  test -f "$result/$label.completed"
  test -s "$result/$label.validated.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.completed"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
