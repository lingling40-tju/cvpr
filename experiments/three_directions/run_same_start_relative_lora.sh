#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in preflight|smoke|full) ;; *) echo 'usage: run_same_start_relative_lora.sh preflight|smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
expanded="$base/policy_action_memory_lora_20261004"
balanced="$base/policy_balanced_change_lora_20261004"
source="$base/policy_control_fit_extension_20261004"
prior="$base/policy_unbounded_expert_cross_goal_potential_lora_20261004"
scratch="$base/policy_same_start_relative_lora_20261004"
run="$scratch/runlogs/$mode"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "same-start relative $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$expanded/runlogs/diversity_fit1024/completed"
test -f "$source/runlogs/fit_extension/completed"
test -f "$source/runlogs/cross_goal_reachable/completed"
test -s "$root/runlogs/ordinal_progress/same_start_pairwise_preflight.json"
if test "$mode" != preflight; then
  test -f "$scratch/runlogs/preflight/completed"
  test -f "$prior/runlogs/full/completed"
  test -f "$prior/runlogs/full/failed_development_gate"
  test -s "$prior/runlogs/full/adapter_head.pt"
  gpu=${VLN_REP_GPU:-0}
  max_used=${VLN_REP_MAX_USED_MIB:-30000}
  [[ "$gpu" =~ ^[0-3]$ ]]
  [[ "$max_used" =~ ^[1-9][0-9]*$ ]]
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n "$((gpu + 1))p" | tr -d ' ')
  test -n "$used" && test "$used" -lt "$max_used"
fi
if test "$mode" = full; then test -f "$scratch/runlogs/smoke/completed"; fi
cd "$root"
export PYTHONPATH="$scratch:$prior:$balanced:$expanded:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
if test "$mode" != preflight; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
python="$base/activevln_train_env/bin/python"
common=(
  --scene-split runlogs/ordinal_progress/stop_history_lora_scene_split.json
  --expert-manifest runlogs/ordinal_progress/stop_history_expert_manifest.json
  --expert-labels runlogs/ordinal_progress/stop_history_label_audit.json
  --expert-root runlogs/ordinal_progress/stop_history_expert_frames
  --policy-manifest runlogs/ordinal_progress/policy_process_manifest.json
  --policy-root runlogs/ordinal_progress/policy_process_turns
  --policy-audit runlogs/ordinal_progress/policy_process_collection/collection_audit.json
  --expanded-manifest "$expanded/diversity_fit1024_manifest.json"
  --expanded-root "$expanded/diversity_fit1024_turns"
  --expanded-audit "$expanded/runlogs/diversity_fit1024/fit_audit.json"
  --render-manifest "$source/render_manifest.json"
  --render-root "$source/render_turns"
  --rgb-audit "$source/runlogs/fit_extension/fit_audit.json"
  --cross-manifest "$source/cross_goal_reachable_manifest.json"
  --cross-root "$source/cross_goal_reachable_labels"
  --cross-audit "$source/runlogs/cross_goal_reachable/audit.json"
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
  --pair-preflight runlogs/ordinal_progress/same_start_pairwise_preflight.json
  --output "$run"
)
if test "$mode" = preflight; then common+=(--preflight); fi
if test "$mode" = smoke; then common+=(--smoke); fi
if test "$mode" != preflight; then
  common+=(--init-checkpoint "$prior/runlogs/full/adapter_head.pt")
fi
sha256sum "$scratch/train_same_start_relative_lora.py" \
  "$prior/train_unbounded_expert_cross_goal_potential_lora.py" \
  "$balanced/train_balanced_change_lora.py" \
  "$expanded/train_policy_progress_lora.py" \
  "$expanded/history_grounding_lora.py" >"$run/source.sha256"
"$python" "$scratch/train_same_start_relative_lora.py" "${common[@]}" \
  >"$run/train.log" 2>&1
if test "$mode" = full; then
  "$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
report=json.load(open(sys.argv[1]));run=pathlib.Path(sys.argv[2])
assert report['schema']=='same_start_relative_lora_development_v1'
name='passed_development_gate' if all(report['development_gate'].values()) else 'failed_development_gate'
(run/name).write_text('scene-disjoint R2R-train development only\n')
PY
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
