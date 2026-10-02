#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
out="$run/geodesic_potential"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock -n 9 || { echo 'geodesic potential screen already running' >&2; exit 2; }
if test -f "$out/suite.completed"; then exit 0; fi
rm -f "$out/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$out/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
for _ in $(seq 1 240); do
  if test -f "$run/geodesic_labels/suite.completed"; then break; fi
  if test -f "$run/geodesic_labels/suite.failed"; then
    echo 'geodesic label replay failed' >&2
    exit 1
  fi
  sleep 30
done
test -f "$run/geodesic_labels/suite.completed"
test -f "$run/navigation_sft_server_prompt.completed"
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/fit_geodesic_potential.py \
  --manifest "$run/manifest.json" \
  --features "$run/navigation_sft_server_prompt/features.pt" \
  --labels-root "$run/geodesic_labels" \
  --output "$out/audit.json" --weights-output "$out/weights.pt" \
  >"$out/fit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/suite.completed"
