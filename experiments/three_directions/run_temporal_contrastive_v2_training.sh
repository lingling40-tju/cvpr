#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
v2="$run/temporal_contrastive_v2"
mkdir -p "$v2"
exec 9>"$v2/training.lock"
flock -n 9 || { echo 'temporal v2 training already running' >&2; exit 2; }
if test -f "$v2/training.completed"; then exit 0; fi
rm -f "$v2/training.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$v2/training.failed"; fi
}
trap on_exit EXIT
cd "$root"
for _ in $(seq 1 480); do
  if test -f "$v2/swaps/fit_development.completed"; then break; fi
  if test -f "$v2/swaps/fit_development.failed"; then
    echo 'wrong-instruction cache failed' >&2
    exit 1
  fi
  sleep 30
done
test -f "$v2/swaps/fit_development.completed"
test -f "$run/geodesic_labels/suite.completed"
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/train_temporal_contrastive_v2.py \
  --manifest "$run/manifest.json" --v2-manifest "$v2/manifest.json" \
  --features "$run/navigation_sft_server_prompt/features.pt" \
  --labels-root "$run/geodesic_labels" --swaps-root "$v2/swaps" \
  --output "$v2/development.json" --checkpoint "$v2/encoder.pt" \
  --seed 11 --epochs 50 >"$v2/train.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$v2/training.completed"
