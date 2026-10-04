#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
scratch="$base/policy_control_fit_extension_20261004"
run="$scratch/runlogs/cross_goal_reachable"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'cross-goal label replay already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$scratch:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_server_env/bin/python"
test "$(sha256sum "$scratch/cross_goal_reachable_manifest.json" | awk '{print $1}')" = \
  1faa89a1284da0d75c9d1b3f785bc89da41dac6abbea06e58daa3e15aa8a7137
test "$(sha256sum "$scratch/render_manifest.json" | awk '{print $1}')" = \
  43bf8bc8af46f07051d299810c1dd2975034a0b84da7f278b98fda5db761006c
"$python" - "$scratch/cross_goal_reachable_smoke/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['requested']==x['completed']==4
assert x['max_correct_distance_drift_m']<=1e-4
assert x['manifest_sha256']=='1faa89a1284da0d75c9d1b3f785bc89da41dac6abbea06e58daa3e15aa8a7137'
PY
sha256sum "$scratch/prepare_cross_goal_fit_manifest.py" \
  "$scratch/resolve_cross_goal_reachability.py" \
  "$scratch/collect_cross_goal_fit_labels.py" \
  "$scratch/audit_cross_goal_fit_labels.py" \
  "$scratch/run_cross_goal_fit_labels.sh" \
  >"$run/source.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
"$python" "$scratch/collect_cross_goal_fit_labels.py" \
  --manifest "$scratch/cross_goal_reachable_manifest.json" \
  --render-manifest "$scratch/render_manifest.json" \
  --render-root "$scratch/render_turns" \
  --output "$scratch/cross_goal_reachable_labels" --gpu 1 \
  >"$run/collect.log" 2>&1
"$python" - "$scratch/cross_goal_reachable_labels/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='cross_goal_fit_label_summary_v1'
assert x['manifest_sha256']=='1faa89a1284da0d75c9d1b3f785bc89da41dac6abbea06e58daa3e15aa8a7137'
assert x['requested']==x['completed']==500 and x['unique_episode_ids']==250
assert x['max_correct_distance_drift_m']<=1e-4 and x['smoke_limit']==0
PY
"$python" "$scratch/audit_cross_goal_fit_labels.py" \
  --ids "$scratch/control_exact512_fit_extension_ids.json" \
  --manifest "$scratch/cross_goal_reachable_manifest.json" \
  --render-root "$scratch/render_turns" \
  --label-root "$scratch/cross_goal_reachable_labels" \
  --output "$run/audit.json" >"$run/audit.log" 2>&1
"$python" - "$scratch/cross_goal_reachable_labels/summary.json" "$run" <<'PY'
import json,sys,time,pathlib
x=json.load(open(sys.argv[1]))
name='passed_sample_gate' if x['sample_gate']['passed'] else 'failed_sample_gate'
pathlib.Path(sys.argv[2],name).write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
