#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_dir="$root/runlogs/multiview_event_source"
rgb="$root/runlogs/multiview_event_rgb/full"
fit="$root/runlogs/multiview_event_lora/full"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
gpu=${VLN_EVENT_FIT_GPU:-1}
[[ "$gpu" =~ ^[0-3]$ ]]
exec 9>"$fit/image_shuffle.lock"
flock -n 9 || { echo 'two-view image shuffle already active' >&2; exit 2; }
test -f "$fit/completed" && test -s "$fit/development.json"
if test -f "$fit/image_shuffle.completed"; then exit 0; fi
rm -f "$fit/image_shuffle.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$fit/image_shuffle.failed"; fi; }
trap on_exit EXIT
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu${gpu}.lock"
exec 8>"$eval_lock"
flock 8
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n "$((gpu + 1))p" | tr -d ' ')
test -n "$used" && test "$used" -lt 5000
export PYTHONPATH="$root/tools:$root:$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
cd "$root"
"$base/activevln_train_env/bin/python" \
  "$root/tools/evaluate_multiview_image_shuffle.py" \
  --capture-manifest "$source_dir/capture_manifest.json" \
  --pair-labels "$source_dir/privileged_pair_labels.json" \
  --verification "$rgb/verification.json" \
  --rgb-root "$rgb/rgb" --model "$model" \
  --checkpoint "$fit/selected.pt" \
  --development "$fit/development.json" \
  --output "$fit/image_shuffle.json" \
  >"$fit/image_shuffle.log" 2>&1
test -s "$fit/image_shuffle.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$fit/image_shuffle.completed"
