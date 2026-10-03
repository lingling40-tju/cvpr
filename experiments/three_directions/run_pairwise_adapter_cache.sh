#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
case "${1:-final}" in
  interim)
    run="$root/runlogs/ordinal_progress/pairwise_interim_collection"
    output="$root/runlogs/ordinal_progress/pairwise_interim_states"
    checkpoint="$root/runlogs/ordinal_progress/history_grounding_lora_seed11/interim_selected.pt"
    ;;
  final)
    run="$root/runlogs/ordinal_progress/pairwise_adapter_collection"
    output="$root/runlogs/ordinal_progress/pairwise_adapter_states"
    checkpoint="$root/runlogs/ordinal_progress/history_grounding_lora_seed11/adapter_head.pt"
    ;;
  *) echo 'usage: run_pairwise_adapter_cache.sh [interim|final]' >&2; exit 2 ;;
esac
manifest="$root/runlogs/ordinal_progress/policy_process_manifest.json"
expected_sha=aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'pairwise adapter cache already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_sha"
test -f "$checkpoint"
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_train_env/bin/python"
common=(
  --scene-split runlogs/ordinal_progress/stop_history_lora_scene_split.json
  --expert-manifest runlogs/ordinal_progress/stop_history_expert_manifest.json
  --expert-labels runlogs/ordinal_progress/stop_history_label_audit.json
  --expert-root runlogs/ordinal_progress/stop_history_expert_frames
  --policy-manifest "$manifest"
  --policy-root runlogs/ordinal_progress/policy_process_turns
  --policy-audit runlogs/ordinal_progress/policy_process_collection/collection_audit.json
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
  --checkpoint "$checkpoint"
  --output-root "$output"
)
collect() {
  local part=$1 shard=$2 shards=$3 gpu=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/cache_pairwise_policy_states.py \
    "${common[@]}" --part "$part" --shard "$shard" --shards "$shards" \
    >"$run/${part}_${shard}.log" 2>&1
}
collect fit 0 2 0 & pid0=$!
collect fit 1 2 1 & pid1=$!
collect development 0 1 2 & pid2=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
test "$status" -eq 0
"$python" tools/audit_pairwise_policy_states.py \
  --policy-manifest "$manifest" --policy-root runlogs/ordinal_progress/policy_process_turns \
  --cache-root "$output" --source-id "$(sha256sum "$checkpoint" | awk '{print $1}')" \
  --output "$run/cache_audit.json" >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
