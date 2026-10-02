#!/usr/bin/env bash
set -euo pipefail

# Start this fallback only if both current optimizer pilots fail the
# predeclared fixed-256 gain gate and their services are released.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
prior="$root/runlogs/optimizer_scale_eval_conditional"
run_dir="$root/runlogs/dynamic_resampling_conditional"
result="$root/runlogs/three_direction_val256"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'dynamic-resampling watcher already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$prior/suite.completed"; do
  if test -f "$prior/suite.failed"; then echo 'optimizer suite failed' >&2; exit 1; fi
  pid_file="$root/runlogs/optimizer_scale_eval_conditional_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "optimizer watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done
if ! test -f "$prior/no_eligible_modes"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test -f "$prior/services.stopped"
cd "$root"
bash tools/start_group4_service.sh 0 >"$run_dir/service.log" 2>&1
bash tools/run_dynamic_resampling_pilot.sh 2 11 >"$run_dir/smoke.launcher.log" 2>&1
test -s "$root/runlogs/three_directions_dynamic_resampling_2step_seed11/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/smoke.completed"
bash tools/run_dynamic_resampling_pilot.sh 64 11 >"$run_dir/train.launcher.log" 2>&1
test -s "$root/runlogs/three_directions_dynamic_resampling_64step_seed11/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train.completed"

checkpoint="$root/verl_checkpoints/three_directions_dynamic_resampling_64step_seed11/global_step_64/actor/huggingface"
test -f "$checkpoint/config.json"
test -f "$result/branch_control64.completed"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
VLN_EVAL_PORT=8016 bash tools/run_direction_eval.sh \
  dynamic64_seed11 "$checkpoint" 3 2 >"$run_dir/eval.launcher.log" 2>&1
test -f "$result/dynamic64_seed11.completed"
analysis="$result/paired_dynamic64_vs_branch_control64.json"
"$base/activevln_server_env/bin/python" tools/analyze_matched_pair.py \
  --root "$result" --candidate dynamic64_seed11 --control branch_control64 \
  --expected-count 256 --output "$analysis" >"$run_dir/analysis.log" 2>&1
test -s "$analysis"
decision=$("$base/activevln_server_env/bin/python" - "$analysis" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1]))
assert d['split']=='val_unseen' and d['episodes']==256 and d['scenes']==11
assert d['manifest_sha256']=='546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46'
assert d['candidate']=='dynamic64_seed11' and d['control']=='branch_control64'
assert d['candidate_metrics']['count']==d['control_metrics']['count']==256
assert d['candidate_metrics']['inference_errors']==d['control_metrics']['inference_errors']==0
sr,spl=d['paired']['sr_pp'],d['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr>0 and spl>=0 else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run_dir/pilot_decision.txt"
if [ "$decision" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_pilot_gain"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
