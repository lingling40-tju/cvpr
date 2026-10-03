#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_fused_reward_20261003"
policy="$source_root/runlogs/ordinal_progress/policy_preference"
run="$root/runlogs/temporal_persistence_probe"
mkdir -p "$run"
exec 9>"$run/probe.lock"
flock -n 9 || { echo 'temporal persistence probe already active' >&2; exit 2; }
if test -f "$run/probe.completed"; then exit 0; fi
rm -f "$run/probe.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/probe.failed"; fi
}
trap on_exit EXIT
test -f "$root/runlogs/failure_rank/cache.completed"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export PYTHONPATH="$root/tools:$source_root/tools:$source_root${PYTHONPATH:+:$PYTHONPATH}"
nice -n 15 "$base/activevln_train_env/bin/python" \
  "$root/tools/probe_temporal_persistence.py" \
  --failure-manifest "$root/runlogs/failure_rank/manifest.json" \
  --failure-features "$root/runlogs/failure_rank/features_full/features.pt" \
  --old-pair-manifest "$policy/manifest.json" \
  --old-v2-manifest "$policy/temporal_contrastive_v2/manifest.json" \
  --old-features "$policy/navigation_sft_server_prompt/features.pt" \
  --old-swaps "$policy/temporal_contrastive_v2/swaps" \
  --old-encoder "$policy/temporal_contrastive_v2/encoder.pt" \
  --new-encoder "$root/runlogs/failure_rank/encoder.pt" \
  --calibration "$root/runlogs/failure_rank/reward_calibration.json" \
  --output "$run/development.json" >"$run/probe.log" 2>&1
"$base/activevln_train_env/bin/python" - "$run/development.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='temporal_persistence_train_scene_screen_v1'
assert x['table']['old_terminal']['development']['failure_near_over_far']['pairs']==125
assert x['table']['old_terminal']['development']['success_over_failure']['pairs']==52
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/probe.completed"
