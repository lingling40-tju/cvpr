#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_dir="$root/runlogs/boundary_occupancy_source"
replay_dir="$root/runlogs/boundary_occupancy_rgb/full"
fit_dir="$root/runlogs/boundary_occupancy_lora/full"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
gpu=${VLN_BOUNDARY_FIT_GPU:-1}
[[ "$gpu" =~ ^[0-3]$ ]]
exec 9>"$fit_dir/shuffle.lock"
flock -n 9 || { echo 'image-shuffle control already active' >&2; exit 2; }
test -f "$fit_dir/completed" && test -s "$fit_dir/development.json"
if test -f "$fit_dir/image_shuffle.completed"; then exit 0; fi
rm -f "$fit_dir/image_shuffle.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$fit_dir/image_shuffle.failed"; fi; }
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
  "$root/tools/evaluate_boundary_image_shuffle.py" \
  --manifest "$source_dir/replay_manifest.json" \
  --labels "$source_dir/privileged_labels.json" \
  --verification "$replay_dir/verification.json" \
  --replay-root "$replay_dir/rgb" --model "$model" \
  --checkpoint "$fit_dir/selected.pt" \
  --development "$fit_dir/development.json" \
  --output "$fit_dir/image_shuffle.json" \
  >"$fit_dir/image_shuffle.log" 2>&1
test -s "$fit_dir/image_shuffle.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$fit_dir/image_shuffle.completed"
