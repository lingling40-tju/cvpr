#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in prompt|smoke|full) ;; *) echo 'usage: run_action_memory_progress_lora.sh prompt|smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
scratch="$base/policy_action_memory_lora_20261004"
run="$scratch/runlogs/$mode"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "action-memory LoRA $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$root/runlogs/ordinal_progress/policy_process_collection/completed"
test "$(sha256sum "$root/runlogs/ordinal_progress/policy_process_manifest.json" | awk '{print $1}')" = \
  aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
if test "$mode" = smoke; then test -f "$scratch/runlogs/prompt/completed"; fi
if test "$mode" = full; then test -f "$scratch/runlogs/smoke/completed"; fi
if test "$mode" != prompt; then
  used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n '2p' | tr -d ' ')
  test -n "$used1" && test "$used1" -lt 8000
fi
cd "$root"
export PYTHONPATH="$scratch:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
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
  --output "$run"
)
if test "$mode" = prompt; then common+=(--prompt-check); fi
if test "$mode" = smoke; then common+=(--smoke); fi
sha256sum "$scratch/train_action_memory_progress_lora.py" >"$run/source.sha256"
"$python" "$scratch/train_action_memory_progress_lora.py" "${common[@]}" \
  >"$run/train.log" 2>&1
if test "$mode" = full; then
  "$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
report=json.load(open(sys.argv[1]))
run=pathlib.Path(sys.argv[2])
assert report['schema']=='action_memory_progress_lora_development_v1'
name='passed_development_gate' if report['full_development_gate'] and all(
    report['full_development_gate'].values()) else 'failed_development_gate'
(run/name).write_text('development-only action-memory reward screen\n')
PY
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
