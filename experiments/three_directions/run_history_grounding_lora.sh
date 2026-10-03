#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/history_grounding_lora_seed11"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
manifest="$root/runlogs/ordinal_progress/policy_process_manifest.json"
expected_sha=aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9 || { echo 'history-grounding LoRA already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_sha"
test -f "$root/runlogs/ordinal_progress/policy_process_collection/completed"
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=3
python="$base/activevln_train_env/bin/python"
common=(
  --scene-split runlogs/ordinal_progress/stop_history_lora_scene_split.json
  --expert-manifest runlogs/ordinal_progress/stop_history_expert_manifest.json
  --expert-labels runlogs/ordinal_progress/stop_history_label_audit.json
  --expert-root runlogs/ordinal_progress/stop_history_expert_frames
  --policy-manifest runlogs/ordinal_progress/policy_process_manifest.json
  --policy-root runlogs/ordinal_progress/policy_process_turns
  --policy-audit runlogs/ordinal_progress/policy_process_collection/collection_audit.json
  --model "$model"
)
if ! test -f "$run/adapter_head.pt" || ! test -f "$run/development.json"; then
  "$python" tools/train_history_grounding_lora.py "${common[@]}" \
    --output "$run" >"$run/train.log" 2>&1
fi
if ! test -f "$run/locked_audit.json"; then
  if ! "$python" tools/audit_history_grounding_lora.py "${common[@]}" \
    --checkpoint "$run/adapter_head.pt" \
    --development "$run/development.json" \
    --output "$run/locked_audit.json" >"$run/audit.log" 2>&1; then
    if grep -q 'development gate failed; locked audit remains unopened' "$run/audit.log"; then
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/development_rejected"
      exit 0
    fi
    exit 1
  fi
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
