#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
run="$root/runlogs/stop_readiness_probe"
manifest="$run/manifest.json"
frames="$run/frames"
mkdir -p "$run"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'STOP probe collection already running' >&2; exit 2; }
if test -f "$run/collection.completed"; then exit 0; fi
rm -f "$run/collection.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/collection.failed"; fi
}
trap on_exit EXIT
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  ec53e58c04e8b1e60a5c19c052ef5d726a44c75ea8ec82d94a70c59c9fcd02ad
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 "$base/activevln_server_env/bin/python" \
  tools/collect_stop_readiness_frames.py --manifest "$manifest" \
  --output "$frames" --gpu 0 >"$run/collection.log" 2>&1
python - "$frames/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['requested_trajectories']==x['completed_trajectories']==531
assert not x['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/collection.completed"
