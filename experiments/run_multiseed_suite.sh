#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930
run_dir="$root/runlogs/eventtrace_r2r64"
mkdir -p "$run_dir"
smoke_checkpoint="$root/verl_checkpoints/eventtrace_seedfix_smoke_seed11_control"
smoke_log="$root/runlogs/eventtrace_seedfix_smoke_control.log"
if [ ! -f "$smoke_checkpoint/global_step_2/data.pt" ]; then
  EVENTTRACE_EXPERIMENT_PREFIX=eventtrace_seedfix_smoke \
    EVENTTRACE_TOTAL_STEPS=2 EVENTTRACE_SAVE_FREQ=2 \
    bash "$root/tools/run_multiseed_train.sh" 11 control >"$smoke_log" 2>&1
fi
python "$root/tools/check_grpo_preflight.py" \
  "$smoke_checkpoint/rollout.jsonl" "$smoke_log" --steps 2 \
  >"$run_dir/preflight.json"
for seed in 11 22 33; do
  for arm in control event; do
    label="seed${seed}_${arm}"
    if [ -f "$run_dir/$label.completed" ]; then continue; fi
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.started"
    rm -f "$run_dir/$label.failed"
    if bash "$root/tools/run_multiseed_train.sh" "$seed" "$arm" \
      >>"$run_dir/$label.log" 2>&1 && \
      python "$root/tools/check_grpo_preflight.py" \
        "$root/verl_checkpoints/eventtrace_r2r64_${label}/rollout.jsonl" \
        "$run_dir/$label.log" --steps 64 \
        >"$run_dir/$label.validated.json" 2>>"$run_dir/$label.log"; then
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.completed"
    else
      status=$?
      printf '%s\n' "$status" >"$run_dir/$label.failed"
      exit "$status"
    fi
  done
done
