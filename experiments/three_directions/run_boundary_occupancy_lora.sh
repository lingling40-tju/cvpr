#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_boundary_occupancy_lora.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_dir="$root/runlogs/boundary_occupancy_source"
replay_dir="$root/runlogs/boundary_occupancy_rgb/full"
scratch="$root/runlogs/boundary_occupancy_lora"
run="$scratch/$mode"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
python="$base/activevln_train_env/bin/python"
gpu=${VLN_BOUNDARY_FIT_GPU:-1}
[[ "$gpu" =~ ^[0-3]$ ]]
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "boundary occupancy LoRA $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$replay_dir/completed" && test -s "$replay_dir/verification.json"
test -s "$source_dir/replay_manifest.json" && test -s "$source_dir/privileged_labels.json"
test -d "$model"
if test "$mode" = full; then
  test -f "$scratch/smoke/completed" && test -s "$scratch/smoke/smoke.json"
fi
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu${gpu}.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n "$((gpu + 1))p" | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU $gpu occupied ($used MiB); defer occupancy fit" >&2; exit 1;
}
export PYTHONPATH="$root/tools:$root:$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
cd "$root"
args=(
  --manifest "$source_dir/replay_manifest.json"
  --labels "$source_dir/privileged_labels.json"
  --verification "$replay_dir/verification.json"
  --replay-root "$replay_dir/rgb"
  --model "$model"
  --output "$run"
)
if test "$mode" = smoke; then args+=(--smoke); fi
"$python" "$root/tools/train_boundary_occupancy_lora.py" "${args[@]}" \
  >"$run/train.log" 2>&1
if test "$mode" = smoke; then
  test -s "$run/smoke.json"
else
  test -s "$run/development.json" && test -s "$run/development_scores.json"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
