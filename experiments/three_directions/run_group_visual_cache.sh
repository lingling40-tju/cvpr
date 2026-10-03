#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
model="$base/models/siglip-base-patch16-224"
run="$root/runlogs/ordinal_progress/group_visual_cache"
output="$root/runlogs/ordinal_progress/group_visual_tokens"
manifest="$root/runlogs/ordinal_progress/group_relative_manifest.json"
replay="$root/runlogs/ordinal_progress/group_relative_collection"
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'group visual cache already running' >&2; exit 2; }
test -f "$replay/completed"
test ! -f "$replay/failed"
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  a99a15020b3d7ffe061ea82ab830e4d617e5ad8f90e3fff8979ab80079a26356
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_train_env/bin/python"
collect() {
  local part=$1 shard=$2 shards=$3 gpu=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/cache_group_visual_tokens.py \
    --manifest "$manifest" \
    --turn-root runlogs/ordinal_progress/group_relative_turns \
    --collection-audit "$replay/collection_audit.json" \
    --model "$model" --part "$part" --shard "$shard" --shards "$shards" \
    --output-root "$output" --batch-size 32 \
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
"$python" tools/audit_group_visual_tokens.py \
  --manifest "$manifest" \
  --turn-root runlogs/ordinal_progress/group_relative_turns \
  --cache-root "$output" \
  --source-id "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" \
  --config-sha "$(sha256sum "$model/config.json" | awk '{print $1}')" \
  --processor-sha "$(sha256sum "$model/preprocessor_config.json" | awk '{print $1}')" \
  --output "$run/cache_audit.json" >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
