#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_fused_reward_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
policy="$source_root/runlogs/ordinal_progress/policy_preference"
scale="$root/runlogs/failure_only_scale"
run="$root/runlogs/failure_visual_fallback"
mkdir -p "$run"
exec 9>"$run/probe.lock"
flock -n 9 || { echo 'failure visual fallback already active' >&2; exit 2; }
if test -f "$run/probe.completed"; then exit 0; fi
rm -f "$run/probe.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/probe.failed"; fi
}
trap on_exit EXIT
until test -f "$scale/suite.completed"; do
  if test -f "$scale/suite.failed"; then echo 'failure-only scale suite failed' >&2; exit 1; fi
  pid=$(cat "$scale/launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'failure-only scale watcher stopped' >&2; exit 1; }
  sleep 60
done
if ! test -f "$scale/no_pilot_gain"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/not_applicable_positive_online_screen"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/probe.completed"
  exit 0
fi
test -f "$root/runlogs/failure_rank/replay.completed"
test -f "$root/runlogs/failure_rank/cache.completed"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$root/tools:$source_root/tools:$source_root${PYTHONPATH:+:$PYTHONPATH}"
nice -n 10 "$base/activevln_train_env/bin/python" \
  "$root/tools/probe_failure_visual_fusion.py" \
  --manifest "$root/runlogs/failure_rank/manifest.json" \
  --replay-root "$root/runlogs/failure_rank/replay_full" \
  --replay-summary "$root/runlogs/failure_rank/replay_full/summary.json" \
  --features "$root/runlogs/failure_rank/features_full/features.pt" \
  --encoder "$root/runlogs/failure_rank/encoder.pt" \
  --calibration "$root/runlogs/failure_rank/reward_calibration.json" \
  --siglip-model "$base/models/siglip-base-patch16-224" \
  --siglip-adapter "$source_root/runlogs/ordinal_progress/siglip_lora_goal_policy_fixed64_768_seed11/adapter.pt" \
  --old-pair-manifest "$policy/manifest.json" \
  --old-v2-manifest "$policy/temporal_contrastive_v2/manifest.json" \
  --output "$run/probe.json" >"$run/probe.log" 2>&1
"$base/activevln_train_env/bin/python" - "$run/probe.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='failure_only_temporal_visual_train_scene_probe_v1'
assert x['fit']['equal_fusion']['pairs']==697
assert x['development']['equal_fusion']['pairs']==125
assert x['audit_opened']==all(x['predeclared_development_gate'].values())
if x['audit_opened']: assert x['audit']['equal_fusion']['pairs']==106
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/probe.completed"
