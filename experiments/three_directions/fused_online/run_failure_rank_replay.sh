#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/failure_rank"
manifest="$run/manifest.json"
output="$run/replay_full"
mkdir -p "$run"
exec 9>"$run/replay.lock"
flock -n 9 || { echo 'failure-rank replay already running' >&2; exit 2; }
if test -f "$run/replay.completed"; then exit 0; fi
rm -f "$run/replay.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/replay.failed"; fi
}
trap on_exit EXIT
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0
if ! test -d "$output" && test -f "$run/replay_smoke/summary.json"; then
  cp "$run/replay_smoke/summary.json" "$run/smoke_summary.json"
  mv "$run/replay_smoke" "$output"
fi
cd "$source_root"
export PYTHONPATH="$source_root/vlnce_server:$source_root${PYTHONPATH:+:$PYTHONPATH}"
nice -n 10 "$base/activevln_server_env/bin/python" \
  "$root/tools/collect_failure_rank_frames.py" \
  --manifest "$manifest" --output "$output" --gpu 1 \
  >"$run/replay.log" 2>&1
"$base/activevln_server_env/bin/python" - "$output/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['pairs'] == 928
assert summary['requested_trajectories'] == summary['completed_trajectories'] == 1856
assert summary['manifest_sha256'] == '139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0'
assert not summary['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/replay.completed"
