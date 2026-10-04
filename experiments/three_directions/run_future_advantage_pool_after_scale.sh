#!/usr/bin/env bash
set -euo pipefail

# CPU-only fallback. Check the already scheduled seed-22 source first;
# use seed 33 only if two audited n=4 sources still miss the frozen gate.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
scale="$root/runlogs/oracle_exact512_scale"
single="$root/runlogs/future_advantage_exact512_preflight"
run="$root/runlogs/future_advantage_pooled_preflight"
dataset="$root/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
split="$base/ActiveVLN_three_directions_20261002/runlogs/ordinal_progress/stop_history_lora_scene_split.json"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'pooled preflight already active' >&2; exit 2; }
if test -f "$run/completed" || test -f "$run/skipped_seed11_passed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

check_suite() {
  test ! -f "$scale/suite.failed" || {
    echo 'n=4 scale failed' >&2; exit 1;
  }
  test -s "$scale/launcher.pid"
  pid=$(cat "$scale/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_exact512_scale_after_full.sh' >/dev/null || {
      echo 'n=4 scale launcher vanished' >&2; exit 1;
    }
}

while ! test -f "$single/completed"; do
  test ! -f "$single/failed" || {
    echo 'seed-11 coverage preflight failed' >&2; exit 1;
  }
  check_suite
  sleep 60
done
test -s "$single/report.json"
if "$base/activevln_train_env/bin/python" - "$single/report.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))[
    "enough_coverage_for_fit_preparation"] else 1)
PY
then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_seed11_passed"
  exit 0
fi

wait_for_seed() {
  local seed=$1
  while ! test -f "$scale/candidate_seed${seed}.completed"; do
    check_suite
    sleep 60
  done
  test -s "$scale/train_audit_seed${seed}.json"
}

run_inventory() {
  local output=$1 log=$2
  shift 2
  local -a runs=(
    --run 11 "$root/verl_checkpoints/oracle_turnwise_exact512_128_seed11/rollout.jsonl"
      "$scale/train_audit_seed11.json"
  )
  local seed
  for seed in "$@"; do
    runs+=(--run "$seed"
      "$root/verl_checkpoints/oracle_turnwise_exact512_128_seed${seed}/rollout.jsonl"
      "$scale/train_audit_seed${seed}.json")
  done
  "$base/activevln_train_env/bin/python" \
    "$root/tools/preflight_group_future_advantage_pool.py" \
    "${runs[@]}" --dataset "$dataset" --scene-split "$split" \
    --output "$output" >"$log" 2>&1
  test -s "$output"
}

passes_gate() {
  "$base/activevln_train_env/bin/python" - "$1" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))[
    "enough_coverage_for_fit_preparation"] else 1)
PY
}

wait_for_seed 22
run_inventory "$run/report_seed11_22.json" "$run/analysis_seed11_22.log" 22
if passes_gate "$run/report_seed11_22.json"; then
  cp "$run/report_seed11_22.json" "$run/report.json"
  printf '11,22\n' >"$run/selected_seeds.txt"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi

wait_for_seed 33
run_inventory "$run/report.json" "$run/analysis.log" 22 33
printf '11,22,33\n' >"$run/selected_seeds.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
