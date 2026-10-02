#!/usr/bin/env bash
set -euo pipefail

# A positive matched 64-step progress pilot triggers three-seed 128-step
# training, exact train-row audits, and fixed-256 plus complete-1839 screens.
# A negative pilot records the decision without consuming more GPU time.
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
progress_root="$base/ActiveVLN_progress_fallback_20261002"
pilot_watcher="$source_root/runlogs/three_direction_progress_fallback_conditional"
pilot_eval="$progress_root/runlogs/progress_pilot_eval"
pilot_analysis="$source_root/runlogs/three_direction_val256/paired_progress64_vs_branch_control64.json"
run_dir="$source_root/runlogs/three_direction_progress_scale_conditional"
screen="$source_root/runlogs/three_direction_val256"
full="$source_root/runlogs/three_direction_full_val_unseen"
python="$base/activevln_server_env/bin/python"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'progress scale watcher already active' >&2; exit 2; }
cd "$source_root"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
rm -f "$run_dir/suite.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

until test -f "$pilot_watcher/watcher.completed"; do
  if test -f "$pilot_watcher/watcher.failed"; then
    echo 'progress pilot watcher failed' >&2; exit 1
  fi
  test -s "$pilot_watcher/watcher.pid"
  pid=$(cat "$pilot_watcher/watcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "pilot watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done

if test -f "$pilot_watcher/branch_gain_retained"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/branch_gain_retained"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test -f "$pilot_eval/suite.completed" && test -s "$pilot_analysis"
eligibility=$("$python" - "$pilot_analysis" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
assert d['split'] == 'val_unseen' and d['episodes'] == 256 and d['scenes'] == 11
assert d['manifest_sha256'] == '546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46'
assert d['candidate'] == 'progress64_seed11' and d['control'] == 'branch_control64'
assert d['candidate_metrics']['count'] == d['control_metrics']['count'] == 256
assert d['candidate_metrics']['inference_errors'] == d['control_metrics']['inference_errors'] == 0
sr, spl = d['paired']['sr_pp'], d['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr > 0 and spl >= 0 else 'ineligible')
PY
)
printf '%s\n' "$eligibility" >"$run_dir/pilot_decision.txt"
if [ "$eligibility" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi
test "$eligibility" = eligible

test -s "$progress_root/progress_source.txt"
bash tools/start_progress_service.sh 3 >"$run_dir/service.log" 2>&1
for seed in 11 22 33; do
  bash tools/run_progress_fallback.sh 128 0,1 "$seed" \
    >"$run_dir/train_seed${seed}.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" tools/audit_progress_training_pair.py \
    --source-root "$source_root" --progress-root "$progress_root" \
    --seed "$seed" --output "$run_dir/train_seed${seed}_pair_audit.json" \
    >"$run_dir/audit_seed${seed}.log" 2>&1
  test -s "$run_dir/train_seed${seed}_pair_audit.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train_seed${seed}.completed"
done

eval_lane() {
  local result_root=$1 count=$2 gpu=$3 port=$4 seed label experiment checkpoint
  shift 4
  for seed in "$@"; do
    label="progress128_seed${seed}"
    experiment=three_directions_progress_fallback_128step
    if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
    checkpoint="$progress_root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
    test -f "$progress_root/runlogs/$experiment/completed"
    test -f "$checkpoint/config.json"
    VLN_EVAL_RESULT_ROOT="$result_root" VLN_EVAL_COUNT="$count" \
      VLN_EVAL_PORT="$port" bash tools/run_direction_eval.sh \
      "$label" "$checkpoint" "$gpu" 2 \
      >"$run_dir/eval_${count}_${label}.launcher.log" 2>&1
    test -f "$result_root/$label.completed"
  done
}

eval_set() {
  local result_root=$1 count=$2 status=0 left right
  eval_lane "$result_root" "$count" 0 8011 11 33 &
  left=$!
  eval_lane "$result_root" "$count" 1 8012 22 &
  right=$!
  wait "$left" || status=1
  wait "$right" || status=1
  test "$status" -eq 0
  "$python" tools/analyze_progress_scaled.py --root "$result_root" \
    --expected-count "$count" \
    --output "$result_root/scale_progress_128_analysis.json" \
    >"$run_dir/analysis_${count}.log" 2>&1
  test -s "$result_root/scale_progress_128_analysis.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/eval_${count}.completed"
}

eval_set "$screen" 256
eval_set "$full" 1839
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
