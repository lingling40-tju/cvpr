#!/usr/bin/env bash
set -euo pipefail

# Train-only goal-view data and frozen-feature screen, resumable per subset.
# No RL policy is launched by this runner.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/goal_views_expanded"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'goal-view expanded screen already running' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  3bb3925a9840aea2dd825c3c201f8d4ceadd8e4949650f534ebf3cda17be76ce
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
model="$base/models/siglip-base-patch16-224"
for subset in calibration fit; do
  case "$subset" in calibration) expected=400;; fit) expected=2755;; esac
  if ! test -f "$run/$subset.collection.completed"; then
    "$base/activevln_server_env/bin/python" tools/collect_goal_views.py \
      --manifest "$run/manifest.json" --subset "$subset" \
      --output "$run" --gpu 0 >"$run/$subset.collection.log" 2>&1
  fi
  "$base/activevln_server_env/bin/python" - "$run/$subset/summary.json" "$expected" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
count=int(sys.argv[2])
assert data['requested']==data['completed']==count and not data['errors']
PY
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/$subset.collection.completed"
  if ! test -f "$run/$subset.features.completed"; then
    "$base/activevln_train_env/bin/python" tools/cache_goal_view_features.py \
      --manifest "$run/manifest.json" --subset "$subset" \
      --records-root "$run" --model "$model" --output "$run/$subset.pt" \
      --batch-size 32 >"$run/$subset.features.log" 2>&1
  fi
  "$base/activevln_train_env/bin/python" - "$run/$subset.pt" "$expected" <<'PY'
import sys,torch
data=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
count=int(sys.argv[2])
assert len(data['episode_ids'])==count
assert data['images'].shape[:2]==(count,4)
assert data['texts'].shape[0]==count
assert torch.isfinite(data['images']).all() and torch.isfinite(data['texts']).all()
PY
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/$subset.features.completed"
done
"$base/activevln_train_env/bin/python" tools/analyze_goal_views.py \
  --manifest "$run/manifest.json" --fit-features "$run/fit.pt" \
  --calibration-features "$run/calibration.pt" \
  --output "$run/raw_audit.json" >"$run/raw_audit.log" 2>&1
test -s "$run/raw_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
