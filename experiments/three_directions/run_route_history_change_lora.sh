#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_route_history_change_lora.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
data="$base/policy_action_memory_lora_20261004"
balanced="$base/policy_balanced_change_lora_20261004"
scratch="$base/policy_route_history_change_lora_20261004"
run="$scratch/runlogs/$mode"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "route-history LoRA $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$data/runlogs/diversity_fit1024/completed"
test -f "$data/runlogs/diversity_fit1024/passed_sample_gate"
test "$(sha256sum "$data/diversity_fit1024_manifest.json" | awk '{print $1}')" = \
  bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37
test "$(sha256sum "$balanced/train_balanced_change_lora.py" | awk '{print $1}')" = \
  461f8337c5ef6ec0262d41b713d4d6fce263dad4e567a94df082cc89eec00507
if test "$mode" = full; then test -f "$scratch/runlogs/smoke/completed"; fi
used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
  sed -n '2p' | tr -d ' ')
test -n "$used1" && test "$used1" -lt 30000
cd "$root"
export PYTHONPATH="$scratch:$balanced:$data:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
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
  --expanded-manifest "$data/diversity_fit1024_manifest.json"
  --expanded-root "$data/diversity_fit1024_turns"
  --expanded-audit "$data/runlogs/diversity_fit1024/fit_audit.json"
  --model "$model"
  --output "$run"
)
if test "$mode" = smoke; then common+=(--smoke); fi
sha256sum "$scratch/train_route_history_change_lora.py" \
  "$balanced/train_balanced_change_lora.py" >"$run/source.sha256"
"$python" "$scratch/train_route_history_change_lora.py" "${common[@]}" \
  >"$run/train.log" 2>&1
if test "$mode" = full; then
  "$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
report=json.load(open(sys.argv[1]));run=pathlib.Path(sys.argv[2])
assert report['schema']=='route_history_change_lora_development_v1'
assert report['source_sha256']['history_prompt']
name='passed_development_gate' if report['full_development_gate'] and all(
    report['full_development_gate'].values()) else 'failed_development_gate'
(run/name).write_text('development-only route-history visual-change probe\n')
PY
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
