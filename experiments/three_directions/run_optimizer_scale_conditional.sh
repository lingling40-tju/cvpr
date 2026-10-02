#!/usr/bin/env bash
set -euo pipefail

mode=${1:?group4 or kl_anchor required}
case "$mode" in group4|kl_anchor) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
screen="$root/runlogs/three_direction_val256"
pilot="$root/runlogs/${mode}_pilot_eval"
pilot_watcher="$root/runlogs/${mode}_eval_watcher"
analysis="$screen/paired_${mode}_64_vs_branch_control64.json"
run_dir="$root/runlogs/${mode}_scale_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo "conditional scale watcher already active: $mode" >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$pilot_watcher/watcher.completed"; do
  if test -f "$pilot_watcher/watcher.failed" || test -f "$pilot/suite.failed"; then
    echo "$mode pilot evaluation failed" >&2
    exit 1
  fi
  pid_file="$root/runlogs/${mode}_eval_watcher_launcher.pid"
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  kill -0 "$pid" 2>/dev/null || { echo "pilot watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done

test -f "$pilot/suite.completed" && test -s "$analysis"
decision=$("$base/activevln_server_env/bin/python" - "$analysis" "$mode" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
mode = sys.argv[2]
assert d['split'] == 'val_unseen' and d['episodes'] == 256 and d['scenes'] == 11
assert d['manifest_sha256'] == '546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46'
assert d['candidate'] == f'{mode}_64_seed11' and d['control'] == 'branch_control64'
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

if [ "$mode" = group4 ]; then
  bash "$root/tools/start_group4_service.sh" 0 >"$run_dir/service.log" 2>&1
else
  bash "$root/tools/start_kl_anchor_service.sh" 1 >"$run_dir/service.log" 2>&1
fi
for seed in 11 22 33; do
  bash "$root/tools/run_optimizer_scale.sh" "$mode" "$seed" \
    >"$run_dir/train_seed${seed}.launcher.log" 2>&1
  experiment="three_directions_${mode}_128step"
  if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
  test -f "$root/runlogs/$experiment/completed"
  test -s "$root/runlogs/$experiment/paired_train_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/training.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
