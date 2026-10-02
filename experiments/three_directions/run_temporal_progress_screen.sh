#!/usr/bin/env bash
set -euo pipefail

# Offline scene-disjoint screen; group size stays four for any later RL test.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
out="$run/temporal_progress"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock -n 9 || { echo 'temporal progress screen already running' >&2; exit 2; }
if test -f "$out/suite.completed"; then exit 0; fi
rm -f "$out/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$out/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$run/geodesic_labels/suite.completed"
test -f "$run/navigation_sft_server_prompt.completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  cc3cb63c0ae1ae3c9b47a9d191dbf548d9dce12e08feed0ec8b314a1b7398c47
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/train_temporal_progress_encoder.py \
  --manifest "$run/manifest.json" \
  --features "$run/navigation_sft_server_prompt/features.pt" \
  --labels-root "$run/geodesic_labels" \
  --output "$out/screen.json" --checkpoint "$out/encoder.pt" \
  --seed 11 --epochs 50 >"$out/train.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/suite.completed"
