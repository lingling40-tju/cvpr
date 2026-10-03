#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/group_relative_cache"
output="$root/runlogs/ordinal_progress/group_relative_states"
manifest="$root/runlogs/ordinal_progress/group_relative_manifest.json"
replay="$root/runlogs/ordinal_progress/group_relative_collection"
checkpoint="$root/runlogs/ordinal_progress/history_grounding_lora_seed11/interim_selected.pt"
expected_sha=a99a15020b3d7ffe061ea82ab830e4d617e5ad8f90e3fff8979ab80079a26356
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'group-relative cache already running' >&2; exit 2; }
test -f "$replay/completed"
test ! -f "$replay/failed"
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
  --manifest "$manifest"
  --turn-root runlogs/ordinal_progress/group_relative_turns
  --collection-audit "$replay/collection_audit.json"
  --old-policy-manifest runlogs/ordinal_progress/policy_process_manifest.json
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
  --checkpoint "$checkpoint"
  --output-root "$output"
)
collect() {
  local part=$1 shard=$2 shards=$3 gpu=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/cache_group_relative_states.py \
    "${common[@]}" --part "$part" --shard "$shard" --shards "$shards" \
    >"$run/${part}_${shard}.log" 2>&1
}
collect fit 0 3 0 & pid0=$!
collect fit 1 3 1 & pid1=$!
collect fit 2 3 2 & pid2=$!
collect development 0 1 3 & pid3=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
wait "$pid3" || status=1
test "$status" -eq 0
"$python" tools/audit_group_relative_states.py \
  --manifest "$manifest" \
  --turn-root runlogs/ordinal_progress/group_relative_turns \
  --cache-root "$output" \
  --source-id "$(sha256sum "$checkpoint" | awk '{print $1}')" \
  --output "$run/cache_audit.json" >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
