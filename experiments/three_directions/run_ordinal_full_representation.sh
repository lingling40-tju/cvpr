#!/usr/bin/env bash
set -euo pipefail

# Wait for train-only frames, then cache SigLIP features once and fit small
# progress heads. This is offline quality control; it never launches RL.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress"
model="$base/models/siglip-base-patch16-224"
mkdir -p "$run"
exec 9>"$run/full_representation.lock"
flock -n 9 || { echo 'ordinal representation pipeline already running' >&2; exit 2; }
if test -f "$run/full_representation.completed"; then exit 0; fi
rm -f "$run/full_representation.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/full_representation.failed"; fi
}
trap on_exit EXIT
cd "$root"
until test -f "$run/full_collection.completed"; do
  if test -f "$run/full_collection.failed"; then echo 'frame collection failed' >&2; exit 1; fi
  pid=$(cat "$run/full_collection.pid")
  kill -0 "$pid" 2>/dev/null || { echo "collector PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -f "$run/full_calibration_collection.completed"
test -f "$run/full_fit_collection.completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  4522b6de453c079d6865d7484d38cdc196ba61103f45f2e79b4e9fcb925f6b65
test -s "$model/model.safetensors"
export CUDA_VISIBLE_DEVICES=0
for subset in fit calibration; do
  output="$run/siglip_full_${subset}.pt"
  if ! test -s "$output"; then
    "$base/activevln_train_env/bin/python" tools/cache_ordinal_progress_features.py \
      --manifest "$run/manifest.json" --subset "$subset" \
      --records-root "$run/frame_collection" --model "$model" \
      --output "$output" --batch-size 12 \
      >"$run/siglip_full_${subset}_cache.log" 2>&1
  fi
  test -s "$output"
done
for seed in 11 22 33; do
  output="$run/siglip_full_start_relative_seed${seed}"
  "$base/activevln_train_env/bin/python" tools/fit_ordinal_progress_head.py \
    --manifest "$run/manifest.json" \
    --fit-features "$run/siglip_full_fit.pt" \
    --calibration-features "$run/siglip_full_calibration.pt" \
    --output-dir "$output" --seed "$seed" --steps 600 \
    --representation start_relative --locked-audit \
    >"$run/siglip_full_seed${seed}.log" 2>&1
  test -s "$output/report.json"
  test -s "$output/head.pt"
done
output="$run/siglip_full_current_only_seed11"
"$base/activevln_train_env/bin/python" tools/fit_ordinal_progress_head.py \
  --manifest "$run/manifest.json" \
  --fit-features "$run/siglip_full_fit.pt" \
  --calibration-features "$run/siglip_full_calibration.pt" \
  --output-dir "$output" --seed 11 --steps 600 \
  --representation current_only --locked-audit \
  >"$run/siglip_full_current_only.log" 2>&1
test -s "$output/report.json"
"$base/activevln_train_env/bin/python" tools/verify_ordinal_full_representation.py \
  --manifest "$run/manifest.json" \
  --fit-features "$run/siglip_full_fit.pt" \
  --calibration-features "$run/siglip_full_calibration.pt" \
  --run-dir "$run" --output "$run/locked_audit_gate.json" \
  >"$run/locked_audit_gate.log" 2>&1
test -s "$run/locked_audit_gate.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/full_representation.completed"
