#!/usr/bin/env bash
set -Eeuo pipefail
root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_direction_20261009
base=/Knowin/foundation/haozhiwang/whz
state="$root/runlogs/sequential_training"
mkdir -p "$state"
exec 9>"$state/sequence.lock"
flock -n 9 || { echo "sequence already active" >&2; exit 2; }
on_exit() { rc=$?; if test "$rc" -ne 0; then printf "%s\n" "$rc" >"$state/sequence.failed"; fi; }
trap on_exit EXIT
rm -f "$state/sequence.failed"
"$base/activevln_train_env/bin/python" "$root/tools/verify_three_direction_freeze.py" "$root" >"$state/freeze_verify.json"
launcher="$root/runlogs/td_grpo_anchor_n4_seed11/train64.launcher.pid"
test -s "$launcher"
pid=$(cat "$launcher")
while kill -0 "$pid" 2>/dev/null; do
  if test -f "$root/runlogs/td_grpo_anchor_n4_seed11/failed"; then
    echo "GRPO anchor training failed" >&2; exit 1
  fi
  if test -f "$root/runlogs/td_grpo_anchor_n4_seed11/completed"; then break; fi
  sleep 20
done
test -f "$root/runlogs/td_grpo_anchor_n4_seed11/completed"
test ! -f "$root/runlogs/td_grpo_anchor_n4_seed11/failed"
for method in turn_rloo srgpo; do
  run="$root/runlogs/td_${method}_n4_seed11"
  test -f "$run/smoke.completed"
  test ! -f "$run/failed"
  "$base/activevln_train_env/bin/python" "$root/tools/verify_three_direction_freeze.py" "$root" >"$state/freeze_verify_${method}.json"
  printf "%s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ') starting $method 64-step continuation" >>"$state/sequence.log"
  bash "$root/tools/run_three_method_train.sh" "$method" 0,1 64 >"$run/train64.launcher.log" 2>&1
  test -f "$run/completed"
  test ! -f "$run/failed"
  test -f "$root/verl_checkpoints/td_${method}_n4_seed11/global_step_64/actor/huggingface/config.json"
  printf "%s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ') completed $method" >>"$state/sequence.log"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"
