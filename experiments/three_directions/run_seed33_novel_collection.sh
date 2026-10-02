#!/usr/bin/env bash
set -euo pipefail

# Episode-disjoint, train-only group-four diagnostic from the completed
# seed-33 rollout. Reuse source checkpoints; keep audit results untouched
# until a candidate and its selection rule are frozen.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference/seed33_novel"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'seed33 novel collection already active' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  11d582ed2b32fad1f6e03b4efa1f46a169a5bdf4ff76dc9eb319865cb4a8c37c
test "$(sha256sum "$model/config.json" | awk '{print $1}')" = \
  9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
server_python="$base/activevln_server_env/bin/python"
train_python="$base/activevln_train_env/bin/python"
"$server_python" tools/collect_policy_preference_frames.py \
  --manifest "$run/manifest.json" --output "$run/frames" --gpu 0 \
  >"$run/collect.log" 2>&1
"$server_python" - "$run/frames/summary.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['pairs']==38 and data['requested_trajectories']==76
assert data['completed_trajectories']==76 and not data['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/frames.completed"
"$train_python" tools/cache_navigation_sft_state.py \
  --manifest "$run/manifest.json" --records-root "$run/frames" \
  --model "$model" --output-root "$run/sft" >"$run/sft.log" 2>&1
"$train_python" - "$run/sft/summary.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['records']==76 and data['frames']==304
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/sft.completed"
"$server_python" tools/label_policy_geodesic_progress.py \
  --manifest "$run/manifest.json" --frames-root "$run/frames" \
  --output-root "$run/geodesic" --gpu 0 >"$run/geodesic.log" 2>&1
"$server_python" - "$run/geodesic/summary.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['requested_trajectories']==76
assert data['completed_trajectories']==76 and not data['errors']
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
