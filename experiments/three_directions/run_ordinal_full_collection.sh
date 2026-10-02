#!/usr/bin/env bash
set -euo pipefail

# Resume the frozen train-only 512/128 frame manifest on one Habitat GPU.
# Do not run a second copy concurrently: each process loads MP3D scenes.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress"
mkdir -p "$run"
exec 9>"$run/full_collection.lock"
flock -n 9 || { echo 'ordinal full collection already running' >&2; exit 2; }
if test -f "$run/full_collection.completed"; then exit 0; fi
rm -f "$run/full_collection.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/full_collection.failed"; fi
}
trap on_exit EXIT
cd "$root"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  4522b6de453c079d6865d7484d38cdc196ba61103f45f2e79b4e9fcb925f6b65
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
for subset in calibration fit; do
  case "$subset" in calibration) expected=128;; fit) expected=512;; esac
  "$base/activevln_server_env/bin/python" tools/collect_ordinal_progress_frames.py \
    --manifest "$run/manifest.json" --subset "$subset" \
    --output "$run/frame_collection" --gpu 0 \
    >"$run/full_${subset}_collection.log" 2>&1
  "$base/activevln_server_env/bin/python" - "$run/frame_collection/$subset/summary.json" "$expected" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert d['requested'] == d['completed'] == expected
assert not d['errors']
assert len(d['episodes']) == len(set(d['episodes'])) == expected
PY
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/full_${subset}_collection.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/full_collection.completed"
