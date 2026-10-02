#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
pilot="$root/runlogs/dynamic_parallel_watcher"
screen="$root/runlogs/three_direction_val256"
full="$root/runlogs/three_direction_full_val_unseen"
run_dir="$root/runlogs/dynamic_scale_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'dynamic scale watcher already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$pilot/watcher.completed"; do
  if test -f "$pilot/watcher.failed"; then echo 'dynamic pilot failed' >&2; exit 1; fi
  pid_file="$root/runlogs/dynamic_parallel_watcher_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "dynamic pilot PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -s "$screen/paired_dynamic64_vs_branch_control64.json"
decision=$("$base/activevln_server_env/bin/python" - "$screen/paired_dynamic64_vs_branch_control64.json" <<'PY'
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
test "$decision" = "$(cat "$pilot/pilot_decision.txt")"
printf '%s\n' "$decision" >"$run_dir/pilot_decision.txt"
if [ "$decision" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$decision" = eligible
cd "$root"
bash tools/start_dynamic_service.sh 1 >"$run_dir/service.log" 2>&1
for seed in 11 22 33; do
  bash tools/run_dynamic_resampling_scale.sh "$seed" \
    >"$run_dir/train_seed${seed}.launcher.log" 2>&1
  experiment=three_directions_dynamic_resampling_128step
  if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
  test -f "$root/runlogs/$experiment/completed"
  test -s "$root/runlogs/$experiment/paired_train_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/training.completed"

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
  echo 'dynamic scale service did not stop' >&2; exit 1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/service.stopped"

for spec in "256:$screen" "1839:$full"; do
  count=${spec%%:*}
  result=${spec#*:}
  if [ "$count" = 256 ]; then
    sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
  else
    sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$sha"
  for seed in 11 22 33; do
    label="dynamic_128_seed${seed}"
    control="branch_control128_seed${seed}"
    experiment=three_directions_dynamic_resampling_128step
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
    test -f "$checkpoint/config.json"
    test -f "$result/$control.completed"
    VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT="$count" VLN_EVAL_PORT=8016 \
      bash tools/run_direction_eval.sh "$label" "$checkpoint" 1 0 \
      >"$run_dir/eval_${count}_${label}.launcher.log" 2>&1
    test -f "$result/$label.completed"
    pair="$result/paired_${label}_vs_${control}.json"
    "$base/activevln_server_env/bin/python" tools/analyze_matched_pair.py \
      --root "$result" --candidate "$label" --control "$control" \
      --expected-count "$count" --output "$pair" \
      >"$run_dir/analyze_${count}_${label}.log" 2>&1
    test -s "$pair"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/eval_${count}_${label}.completed"
  done
  "$base/activevln_server_env/bin/python" tools/analyze_optimizer_scaled.py \
    --mode dynamic --root "$result" --count "$count" \
    --output "$result/scale_dynamic_128_analysis.json" \
    >"$run_dir/aggregate_${count}.log" 2>&1
  test -s "$result/scale_dynamic_128_analysis.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/aggregate_${count}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
