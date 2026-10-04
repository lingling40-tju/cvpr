#!/usr/bin/env bash
set -euo pipefail

# Diagnose the selected representation on a small *fit* subset only after
# its frozen development test fails. This does not open audit or online RL.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
expanded="$base/policy_action_memory_lora_20261004"
balanced="$base/policy_balanced_change_lora_20261004"
source="$base/policy_control_fit_extension_20261004"
scratch="$base/policy_unbounded_expert_cross_goal_potential_lora_20261004"
full="$scratch/runlogs/full"
run="$scratch/runlogs/fit_diagnostic"
scale="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_scale"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'fit diagnostic already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
while ! test -f "$full/completed"; do
  test ! -f "$full/failed" || { echo 'representation fit failed mechanically' >&2; exit 1; }
  test -s "$scratch/runlogs/after_unbounded/launcher.pid"
  pid=$(cat "$scratch/runlogs/after_unbounded/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_unbounded_expert_after_unbounded.sh' >/dev/null || {
      echo 'representation launcher vanished' >&2; exit 1;
    }
  sleep 30
done
if test -f "$full/passed_development_gate"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_after_pass"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
test -f "$full/failed_development_gate"
test -s "$full/adapter_head.pt" && test -s "$full/development.json"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n '1p' | tr -d ' ')
  test -n "$used"
  if test "$used" -lt 30000; then break; fi
  sleep 30
done
if ! test -f "$scale/training.completed"; then
  curl -fsS --max-time 3 http://127.0.0.1:5013/health \
    >"$run/habitat_health_before.json"
fi
cd "$root"
export PYTHONPATH="$scratch:$balanced:$expanded:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
sha256sum "$scratch/diagnose_unbounded_expert_fit.py" >"$run/source.sha256"
"$base/activevln_train_env/bin/python" \
  "$scratch/diagnose_unbounded_expert_fit.py" \
  --scene-split runlogs/ordinal_progress/stop_history_lora_scene_split.json \
  --expert-manifest runlogs/ordinal_progress/stop_history_expert_manifest.json \
  --expert-labels runlogs/ordinal_progress/stop_history_label_audit.json \
  --expert-root runlogs/ordinal_progress/stop_history_expert_frames \
  --policy-manifest runlogs/ordinal_progress/policy_process_manifest.json \
  --policy-root runlogs/ordinal_progress/policy_process_turns \
  --policy-audit runlogs/ordinal_progress/policy_process_collection/collection_audit.json \
  --expanded-manifest "$expanded/diversity_fit1024_manifest.json" \
  --expanded-root "$expanded/diversity_fit1024_turns" \
  --expanded-audit "$expanded/runlogs/diversity_fit1024/fit_audit.json" \
  --render-manifest "$source/render_manifest.json" \
  --render-root "$source/render_turns" \
  --rgb-audit "$source/runlogs/fit_extension/fit_audit.json" \
  --cross-manifest "$source/cross_goal_reachable_manifest.json" \
  --cross-root "$source/cross_goal_reachable_labels" \
  --cross-audit "$source/runlogs/cross_goal_reachable/audit.json" \
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --fit-output "$full" --output "$run/fit_vs_development.json" \
  >"$run/diagnose.log" 2>&1
test -s "$run/fit_vs_development.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
