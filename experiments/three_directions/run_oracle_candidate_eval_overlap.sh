#!/usr/bin/env bash
set -euo pipefail

# Continue the GPU-1 evaluation lane after the trained controls. The
# group-four candidate keeps training on GPUs 2/3 while its completed
# earlier seeds are evaluated on the frozen full val-unseen manifest.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
scale="$root/runlogs/oracle_exact512_scale"
result="$root/runlogs/oracle_exact512_full1839"
control_eval="$root/runlogs/oracle_control_eval_overlap"
run="$root/runlogs/oracle_candidate_eval_overlap"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'candidate full-eval overlap already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$control_eval/completed"; do
  test ! -f "$control_eval/failed" || {
    echo 'control full-eval overlap failed' >&2; exit 1;
  }
  test -s "$control_eval/launcher.pid"
  pid=$(cat "$control_eval/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_control_eval_overlap.sh' >/dev/null || {
      echo 'control full-eval overlap launcher vanished' >&2; exit 1;
    }
  sleep 30
done
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e

for seed in 11 22 33; do
  label="oracle_turnwise_exact512_128_seed${seed}"
  checkpoint="$root/verl_checkpoints/$label/global_step_128/actor/huggingface"
  while ! test -f "$scale/candidate_seed${seed}.completed"; do
    test ! -f "$scale/suite.failed" || { echo 'n=4 scale failed' >&2; exit 1; }
    test -s "$scale/launcher.pid"
    pid=$(cat "$scale/launcher.pid")
    ps -o args= -p "$pid" | grep -F \
      'run_oracle_exact512_scale_after_full.sh' >/dev/null || {
        echo 'n=4 scale launcher vanished' >&2; exit 1;
      }
    sleep 30
  done
  test -f "$checkpoint/config.json"
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT=8122 \
    VLN_VLLM_SEED="$seed" bash "$control/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" 1 0 \
      >"$run/$label.launcher.log" 2>&1
  test -f "$result/$label.completed"
  test -s "$result/$label.validated.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/$label.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
