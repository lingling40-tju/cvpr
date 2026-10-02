#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
parent="$root/runlogs/ordinal_progress/policy_preference"
run="$parent/seed33_novel"
mkdir -p "$run"
exec 9>"$run/check.lock"
flock -n 9 || { echo 'seed33 novel representation check already running' >&2; exit 2; }
if test -f "$run/check.completed"; then exit 0; fi
rm -f "$run/check.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/check.failed"; fi
}
trap on_exit EXIT
cd "$root"
for _ in $(seq 1 240); do
  if test -f "$run/swaps/suite.completed"; then break; fi
  if test -f "$run/swaps/suite.failed" || test -f "$run/suite.failed"; then
    echo 'novel input preparation failed' >&2
    exit 1
  fi
  sleep 30
done
test -f "$run/suite.completed"
test -f "$run/swaps/suite.completed"
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/evaluate_seed33_novel_representation.py \
  --manifest "$run/manifest.json" --features "$run/sft/features.pt" \
  --labels-root "$run/geodesic" --swaps-manifest "$run/instruction_swaps.json" \
  --swaps-root "$run/swaps" \
  --encoder-v1 "$parent/temporal_progress/encoder.pt" \
  --encoder-v2 "$parent/temporal_contrastive_v2/encoder.pt" \
  --output "$run/representation_check.json" >"$run/check.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/check.completed"
