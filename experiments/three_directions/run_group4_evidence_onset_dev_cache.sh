#!/usr/bin/env bash
set -euo pipefail

kind=${1:?expected policy or expert}
case "$kind" in policy|expert) ;; *) echo 'bad cache kind' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/group4_evidence_onset_lora_seed11"
state_root="$run/${kind}_states"
mkdir -p "$state_root"
exec 9>"$state_root/cache.lock"
flock -n 9 || { echo "cache $kind already running" >&2; exit 2; }
test -f "$run/completed"
test -f "$run/final_adapter.pt"
if test -f "$state_root/completed"; then exit 0; fi
rm -f "$state_root/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state_root/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_train_env/bin/python"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
if test "$kind" = policy; then
  manifest=runlogs/ordinal_progress/group_relative_manifest.json
  record_root=runlogs/ordinal_progress/group_relative_turns
  shards=1
else
  manifest=runlogs/ordinal_progress/group4_joint_value_expert_manifest.json
  record_root=runlogs/ordinal_progress/stop_history_expert_frames
  shards=3
fi
cache() {
  local shard=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/cache_group4_prefix_contrast_dev.py \
    --kind "$kind" --manifest "$manifest" --record-root "$record_root" \
    --model "$model" --checkpoint "$run/final_adapter.pt" \
    --evidence-manifest runlogs/ordinal_progress/group4_evidence_onset_manifest.json \
    --shard "$shard" --shards "$shards" --output-root "$state_root" \
    >"$state_root/shard${shard}.log" 2>&1
}
if test "$kind" = policy; then
  cache 0 0
else
  cache 0 1 & pid0=$!
  cache 1 2 & pid1=$!
  cache 2 3 & pid2=$!
  status=0
  wait "$pid0" || status=1
  wait "$pid1" || status=1
  wait "$pid2" || status=1
  test "$status" -eq 0
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state_root/completed"
