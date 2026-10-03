#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/group4_evidence_onset_lora_seed11"
manifest="$root/runlogs/ordinal_progress/group4_evidence_onset_manifest.json"
expected=41b465132819ae6060d06b3cf0a6ff964d9be83d3ace265ae39414aa1258f26f
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9 || { echo 'evidence-onset training already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected"
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
python="$base/activevln_train_env/bin/python"
common=(
  --group-manifest runlogs/ordinal_progress/group_relative_manifest.json
  --group-collection-audit runlogs/ordinal_progress/group_relative_collection/collection_audit.json
  --group-state-audit runlogs/ordinal_progress/group_relative_cache/cache_audit.json
  --group-turn-root runlogs/ordinal_progress/group_relative_turns
  --expert-manifest runlogs/ordinal_progress/group4_joint_value_expert_manifest.json
  --evidence-manifest "$manifest"
  --expert-prefix-audit runlogs/ordinal_progress/group4_expert_prefix_states/cache_audit.json
  --expert-root runlogs/ordinal_progress/stop_history_expert_frames
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
  --initial-checkpoint runlogs/ordinal_progress/history_grounding_lora_seed11/interim_selected.pt
  --frozen-weights runlogs/ordinal_progress/group4_joint_value_development.pt
)
if ! test -f "$run/smoke/training.json"; then
  "$python" tools/train_group4_evidence_onset_lora.py "${common[@]}" \
    --smoke --output "$run/smoke" >"$run/smoke.log" 2>&1
fi
if ! test -f "$run/final_adapter.pt" || ! test -f "$run/training.json"; then
  "$python" tools/train_group4_evidence_onset_lora.py "${common[@]}" \
    --output "$run" >"$run/train.log" 2>&1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
