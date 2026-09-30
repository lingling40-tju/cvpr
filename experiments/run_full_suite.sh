#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
train_log="$root/runlogs/eventtrace_r2r64"
eval_log="$root/runlogs/eventtrace_full_val_unseen"
mkdir -p "$eval_log"

while [ ! -f "$train_log/seed33_event.completed" ]; do
  if find "$train_log" -maxdepth 1 -name '*.failed' | grep -q .; then
    echo "training failed; full evaluation cannot start" >&2
    exit 1
  fi
  sleep 60
done

for label in sft seed11_control seed11_event seed22_control seed22_event seed33_control seed33_event; do
  case "$label" in
    sft) model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" ;;
    *) model="$root/verl_checkpoints/eventtrace_r2r64_${label}/global_step_64/actor/huggingface" ;;
  esac
  if [ -f "$eval_log/$label.completed" ]; then continue; fi
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$eval_log/$label.started"
  if bash "$root/tools/run_full_val_unseen.sh" "$label" "$model" \
    >"$eval_log/$label.run.log" 2>&1; then
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$eval_log/$label.completed"
  else
    status=$?
    printf '%s\n' "$status" >"$eval_log/$label.failed"
    exit "$status"
  fi
done
"$base/activevln_server_env/bin/python" "$root/tools/analyze_full_val.py" \
  >"$eval_log/analysis.log" 2>&1
