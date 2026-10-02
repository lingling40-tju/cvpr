#!/usr/bin/env bash
set -euo pipefail

# Predeclared scale gate for the compute-matched 2+2 grouping. It uses the
# same fixed-screen SR/SPL criterion as the four-way pilot and waits until
# early group4 evaluation has released GPU 0/1 before starting training.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_group4_pairwise_20261002"
source_root="$base/ActiveVLN_three_directions_20261002"
pilot="$root/runlogs/group4_pairwise_64pilot"
early="$source_root/runlogs/group4_early_eval_after_pairwise"
screen="$source_root/runlogs/three_direction_val256"
group4_gate="$source_root/runlogs/group4_scale_conditional"
run_dir="$root/runlogs/group4_pairwise_scale_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'pairwise scale watcher already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"

stop_service() {
  pid_file="$root/runlogs/group4_pairwise_service/server.pid"
  if test -s "$pid_file"; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      ps -o args= -p "$pid" | grep -F 'server.port=5017' >/dev/null || return 1
      kill "$pid"
    fi
  fi
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5017/health >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  ! curl -fsS --max-time 2 http://127.0.0.1:5017/health >/dev/null 2>&1
}
service_started=0
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    if [ "$service_started" -eq 1 ]; then stop_service || true; fi
    printf '%s\n' "$status" >"$run_dir/suite.failed"
  fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$pilot/watcher.completed"; do
  if test -f "$pilot/watcher.failed"; then echo 'pairwise pilot failed' >&2; exit 1; fi
  pid_file="$root/runlogs/group4_pairwise_64pilot.launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "pilot PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -f "$pilot/service.stopped"
test -f "$pilot/eval.completed"
test "$(sha256sum "$screen/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
analysis="$screen/paired_group4_pairwise64_vs_branch_control64.json"
test -s "$analysis"
decision=$("$base/activevln_server_env/bin/python" - "$analysis" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
assert d['split'] == 'val_unseen' and d['episodes'] == 256 and d['scenes'] == 11
assert d['manifest_sha256'] == '546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46'
assert d['candidate'] == 'group4_pairwise64_seed11' and d['control'] == 'branch_control64'
assert d['candidate_metrics']['count'] == d['control_metrics']['count'] == 256
assert d['candidate_metrics']['inference_errors'] == d['control_metrics']['inference_errors'] == 0
sr, spl = d['paired']['sr_pp'], d['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr > 0 and spl >= 0 else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run_dir/pilot_decision.txt"
if [ "$decision" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$decision" = eligible

until test -f "$early/watcher.completed"; do
  if test -f "$early/watcher.failed"; then echo 'early group4 evaluation failed' >&2; exit 1; fi
  pid_file="$source_root/runlogs/group4_early_eval_after_pairwise_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "early evaluator PID $pid stopped" >&2; exit 1; }
  sleep 60
done

service_started=1
bash "$root/tools/start_group4_pairwise_service.sh" 1 >"$run_dir/service.log" 2>&1
for seed in 11 22 33; do
  source_run="three_directions_group4_128step"
  if [ "$seed" != 11 ]; then source_run="${source_run}_seed${seed}"; fi
  until test -f "$source_root/runlogs/$source_run/completed"; do
    if test -f "$group4_gate/suite.failed"; then echo 'four-way scale failed' >&2; exit 1; fi
    pid_file="$source_root/runlogs/group4_scale_conditional_launcher.pid"
    test -s "$pid_file"
    pid=$(cat "$pid_file")
    kill -0 "$pid" 2>/dev/null || { echo "four-way scale PID $pid stopped" >&2; exit 1; }
    sleep 60
  done
  VLN_PAIRWISE_GPUS=0,1 bash "$root/tools/run_group4_pairwise_training.sh" 128 "$seed" \
    >"$run_dir/train_seed${seed}.launcher.log" 2>&1
  name="three_directions_group4_pairwise_128step_seed${seed}"
  test -f "$root/runlogs/$name/completed"
  test -s "$root/runlogs/$name/paired_train_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/training.completed"
stop_service
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/service.stopped"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
