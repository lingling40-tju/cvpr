#!/usr/bin/env bash
set -euo pipefail

# Use the released GPU 0/1 lane while four-sample scale trains on GPU 2/3.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
smoke="$root/runlogs/three_directions_dynamic_resampling_2step_seed11"
training="$root/runlogs/three_directions_dynamic_resampling_64step_seed11"
run_dir="$root/runlogs/dynamic_parallel_watcher"
result="$root/runlogs/three_direction_val256"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'dynamic parallel watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/watcher.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"
cd "$root"

until test -f "$smoke/completed"; do
  if test -f "$smoke/failed"; then echo 'dynamic smoke failed' >&2; exit 1; fi
  pid_file="$root/runlogs/dynamic_smoke_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "smoke PID $pid stopped" >&2; exit 1; }
  sleep 30
done
test -s "$smoke/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/smoke.completed"

VLN_DYNAMIC_SERVICE_URL=http://127.0.0.1:5015 VLN_DYNAMIC_GPUS=0,1 \
  bash tools/run_dynamic_resampling_pilot.sh 64 11 >"$run_dir/train.launcher.log" 2>&1
test -f "$training/completed" && test -s "$training/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train.completed"

# Model inference uses GPU 1, so release its dedicated training simulator.
pid_file="$root/runlogs/dynamic_service/server.pid"
test -s "$pid_file"
pid=$(cat "$pid_file")
ps -o args= -p "$pid" | grep -F 'server.port=5015' >/dev/null
kill "$pid"
for attempt in $(seq 1 30); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5015/health >/dev/null 2>&1; then break; fi
  sleep 1
done
if curl -fsS --max-time 2 http://127.0.0.1:5015/health >/dev/null 2>&1; then
  echo 'dynamic training service did not stop' >&2; exit 1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/service.stopped"

checkpoint="$root/verl_checkpoints/three_directions_dynamic_resampling_64step_seed11/global_step_64/actor/huggingface"
test -f "$checkpoint/config.json"
test -f "$result/branch_control64.completed"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
VLN_EVAL_PORT=8016 bash tools/run_direction_eval.sh \
  dynamic64_seed11 "$checkpoint" 1 0 >"$run_dir/eval.launcher.log" 2>&1
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
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
