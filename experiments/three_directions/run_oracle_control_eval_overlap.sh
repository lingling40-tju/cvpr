#!/usr/bin/env bash
set -euo pipefail

# Evaluate already completed n=4 controls on GPU 1 while the matched
# candidate/control trainer uses GPUs 2/3. The existing full-suite evaluator
# skips exact labels that this script has validated and completed. No model
# selection is made from these control-only results.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
fit="$base/policy_cross_goal_potential_lora_20261004/runlogs/full"
scale="$root/runlogs/oracle_exact512_scale"
result="$root/runlogs/oracle_exact512_full1839"
run="$root/runlogs/oracle_control_eval_overlap"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'control full-eval overlap already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

# Do not run a vLLM server alongside the full-history LoRA on GPU 1.
while ! test -f "$fit/completed" && ! test -f "$fit/failed"; do
  test ! -f "$scale/suite.failed" || { echo 'n=4 scale failed' >&2; exit 1; }
  if test -s "$fit/launcher.pid"; then
    pid=$(cat "$fit/launcher.pid")
    ps -o args= -p "$pid" | grep -F 'run_cross_goal_potential_lora.sh full' \
      >/dev/null || { echo 'representation fit vanished' >&2; exit 1; }
  fi
  sleep 30
done
while true; do
  used1=$(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
  test -n "$used1"
  if test "$used1" -lt 8000; then break; fi
  sleep 30
done

mkdir -p "$result"
source_manifest="$control/runlogs/three_direction_full_val_unseen/manifest.json"
test "$(sha256sum "$source_manifest" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
if test ! -e "$result/manifest.json"; then cp "$source_manifest" "$result/manifest.json"; fi
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e

for seed in 11 22 33; do
  label="oracle_exact512_control_128_seed${seed}"
  checkpoint="$control/verl_checkpoints/$label/global_step_128/actor/huggingface"
  while ! test -f "$scale/control_seed${seed}.completed"; do
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
