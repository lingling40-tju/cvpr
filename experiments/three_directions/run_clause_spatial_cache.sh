#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
python="$base/activevln_train_env/bin/python"
source_root="$root/runlogs/ordinal_progress/frame_collection"
output="$root/runlogs/ordinal_progress/clause_alignment/spatial_tokens"
manifest="$root/runlogs/ordinal_progress/clause_alignment/manifest.json"
model="$base/models/siglip-base-patch16-224"
mkdir -p "$output"

for gpu in 0 1 2 3; do
  if [[ "$gpu" == 3 ]]; then
    part=calibration
    shard=0
    shards=1
  else
    part=fit
    shard="$gpu"
    shards=3
  fi
  log="$output/${part}_${shard}_of_${shards}.log"
  pid="$output/${part}_${shard}_of_${shards}.pid"
  if [[ -e "$pid" ]] && kill -0 "$(cat "$pid")" 2>/dev/null; then
    echo "already running: $pid"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/tools:$root/vlnce_server:$root" \
    nohup "$python" "$root/tools/cache_clause_spatial_features.py" \
      --manifest "$manifest" --record-root "$source_root" --model "$model" \
      --part "$part" --shard "$shard" --shards "$shards" \
      --output-root "$output" >"$log" 2>&1 </dev/null &
  echo "$!" >"$pid"
  echo "started $part shard $shard/$shards pid $(cat "$pid")"
done
