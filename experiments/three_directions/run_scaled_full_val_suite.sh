#!/usr/bin/env bash
set -euo pipefail

mode=${1:?branch, recovery, or counterfactual required}
steps=${2:-128}
case "$mode" in branch|recovery|counterfactual) ;; *) echo "unknown mode: $mode" >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
screen="$root/runlogs/three_direction_scale_${mode}_${steps}step_eval"
pilot="$root/runlogs/three_direction_val256"
result="$root/runlogs/three_direction_full_val_unseen"
run_dir="$root/runlogs/three_direction_scale_${mode}_${steps}step_full_eval"
mkdir -p "$run_dir" "$result"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'full evaluation already active' >&2; exit 2; }
cd "$root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"
test -f "$result/manifest.json"

until test -f "$screen/suite.completed"; do
  if test -f "$screen/suite.failed"; then echo 'scaled 256 evaluation failed' >&2; exit 1; fi
  pid=$(cat "$screen/suite.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "scaled evaluation PID $pid stopped" >&2; exit 1; fi
  sleep 60
done
if test -f "$screen/not_eligible"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/not_eligible"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
replication=$("$base/activevln_server_env/bin/python" - "$mode" "$steps" <<'PY'
import json, sys
from pathlib import Path
mode, steps = sys.argv[1:]
root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val256')
data = json.loads((root / f'scale_{mode}_{steps}_analysis.json').read_text())
seeds = data['paired_seed_differences']
positive = sum(seeds[str(seed)]['sr_pp'] > 0 for seed in (11, 22, 33))
passed = (data['mean_paired_sr_pp'] > 0 and data['mean_paired_spl_pp'] >= 0
          and positive >= 2)
print('replicate' if passed else 'no_replication')
PY
)
if [ "$replication" = no_replication ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_replication_signal"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$replication" = replicate

evaluate_lane() {
  local arm=$1 gpu=$2 port=$3 seed label experiment checkpoint
  for seed in 11 22 33; do
    label="${arm}${steps}_seed${seed}"
    experiment="three_directions_${arm}_${steps}step"
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    checkpoint="$root/verl_checkpoints/$experiment/global_step_${steps}/actor/huggingface"
    test -f "$root/runlogs/$experiment/completed"
    test -f "$checkpoint/config.json"
    VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT=1839 VLN_EVAL_PORT="$port" \
      bash tools/run_direction_eval.sh "$label" "$checkpoint" "$gpu" 2 \
      >"$run_dir/$label.launcher.log" 2>&1
    test -f "$result/$label.completed"
  done
}

evaluate_lane "$mode" 1 8011 &
candidate_pid=$!
evaluate_lane "${mode}_control" 3 8012 &
control_pid=$!
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0

"$base/activevln_server_env/bin/python" tools/analyze_scaled_val256.py \
  "$mode" --steps "$steps" --root "$result" --expected-count 1839 \
  >"$run_dir/analysis.log" 2>&1
test -s "$result/scale_${mode}_${steps}_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
