#!/usr/bin/env bash
set -euo pipefail

# Replay train-only successful/failed group-four trajectories, no RL update.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
output="$run/frames_full"
mkdir -p "$run"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'policy preference replay already running' >&2; exit 2; }
if test -f "$run/collection.completed"; then exit 0; fi
rm -f "$run/collection.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/collection.failed"; fi
}
trap on_exit EXIT
cd "$root"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  cc3cb63c0ae1ae3c9b47a9d191dbf548d9dce12e08feed0ec8b314a1b7398c47
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
"$base/activevln_server_env/bin/python" tools/collect_policy_preference_frames.py \
  --manifest "$run/manifest.json" --output "$output" --gpu 0 \
  >"$run/collection.log" 2>&1
"$base/activevln_server_env/bin/python" - "$output/summary.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1]))
assert summary['pairs']==400
assert summary['requested_trajectories']==summary['completed_trajectories']==800
assert not summary['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/collection.completed"
