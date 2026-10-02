#!/usr/bin/env bash
set -euo pipefail

# Train-only privileged labels for an offline potential screen. No RL or
# validation-set geodesic labels are read by the candidate reward.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
output="$run/geodesic_labels"
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { echo 'geodesic label replay already running' >&2; exit 2; }
if test -f "$output/suite.completed"; then exit 0; fi
rm -f "$output/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$output/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$run/collection.completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  cc3cb63c0ae1ae3c9b47a9d191dbf548d9dce12e08feed0ec8b314a1b7398c47
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
"$base/activevln_server_env/bin/python" tools/label_policy_geodesic_progress.py \
  --manifest "$run/manifest.json" --frames-root "$run/frames_full" \
  --output-root "$output" --gpu 0 >"$output/replay.log" 2>&1
"$base/activevln_server_env/bin/python" - "$output/summary.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['requested_trajectories']==800
assert report['completed_trajectories']==800 and not report['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$output/suite.completed"
