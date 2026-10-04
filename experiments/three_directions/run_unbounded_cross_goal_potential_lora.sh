#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in preflight|smoke|full) ;; *) echo 'usage: run_unbounded_cross_goal_potential_lora.sh preflight|smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
expanded="$base/policy_action_memory_lora_20261004"
balanced="$base/policy_balanced_change_lora_20261004"
source="$base/policy_control_fit_extension_20261004"
scratch="$base/policy_unbounded_cross_goal_potential_lora_20261004"
run="$scratch/runlogs/$mode"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "unbounded cross-goal potential $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$expanded/runlogs/diversity_fit1024/completed"
test -f "$source/runlogs/fit_extension/completed"
test -f "$source/runlogs/fit_extension/passed_sample_gate"
test -f "$source/runlogs/cross_goal_reachable/completed"
test -f "$source/runlogs/cross_goal_reachable/passed_sample_gate"
test "$(sha256sum "$source/render_manifest.json" | awk '{print $1}')" = \
  43bf8bc8af46f07051d299810c1dd2975034a0b84da7f278b98fda5db761006c
test "$(sha256sum "$source/cross_goal_reachable_manifest.json" | awk '{print $1}')" = \
  1faa89a1284da0d75c9d1b3f785bc89da41dac6abbea06e58daa3e15aa8a7137
test "$(sha256sum "$source/runlogs/cross_goal_reachable/audit.json" | awk '{print $1}')" = \
  03eaeee0991c695dc455f71cd8577ba80f2df7d72fdd2e99acde5970c46a609d
if test "$mode" = smoke; then test -f "$scratch/runlogs/preflight/completed"; fi
if test "$mode" = full; then test -f "$scratch/runlogs/smoke/completed"; fi
gpu=${VLN_REP_GPU:-1}
max_used=${VLN_REP_MAX_USED_MIB:-8000}
[[ "$gpu" =~ ^[0-3]$ ]]
[[ "$max_used" =~ ^[1-9][0-9]*$ ]]
if test "$mode" != preflight; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n "$((gpu + 1))p" | tr -d ' ')
  test -n "$used" && test "$used" -lt "$max_used"
fi
cd "$root"
export PYTHONPATH="$scratch:$balanced:$expanded:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"
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
  --model "$model"
  --output "$run"
)
if test "$mode" = smoke; then common+=(--smoke); fi
if test "$mode" = preflight; then common+=(--preflight); fi
sha256sum "$scratch/train_unbounded_cross_goal_potential_lora.py" \
  "$balanced/train_balanced_change_lora.py" \
  "$expanded/train_policy_progress_lora.py" \
  "$expanded/history_grounding_lora.py" >"$run/source.sha256"
"$python" "$scratch/train_unbounded_cross_goal_potential_lora.py" "${common[@]}" \
  >"$run/train.log" 2>&1
if test "$mode" = full; then
  "$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
report=json.load(open(sys.argv[1]));run=pathlib.Path(sys.argv[2])
assert report['schema']=='unbounded_cross_goal_potential_lora_development_v1'
name='passed_development_gate' if report['full_development_gate'] and all(
    report['full_development_gate'].values()) else 'failed_development_gate'
(run/name).write_text('development-only unbounded crossed-goal potential probe\n')
PY
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
