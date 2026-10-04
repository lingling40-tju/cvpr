#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
scratch="$root/runlogs/early_anchor_audit"
run="$scratch/scoring"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
checkpoint="$root/runlogs/anchor_distance_potential_lora/full/adapter_head.pt"
python="$base/activevln_train_env/bin/python"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'audit scoring already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$scratch/full/completed"
test -s "$scratch/full/verification.json"
test "$(sha256sum "$scratch/source_manifest.json" | awk '{print $1}')" = \
  dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = \
  0b955f0cde2d77a89f48f0d72e2346c65afaf1b46bd1578d716caf4ae5ef3729
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu1.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 occupied ($used MiB)" >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
"$python" tools/score_early_anchor_audit.py \
  --manifest "$scratch/source_manifest.json" \
  --rgb-root "$scratch/rgb" --model "$model" \
  --checkpoint "$checkpoint" --output "$scratch/scores.jsonl" \
  >"$run/score.log" 2>&1
test -s "$scratch/scores.jsonl"
test -s "$scratch/scores.summary.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
