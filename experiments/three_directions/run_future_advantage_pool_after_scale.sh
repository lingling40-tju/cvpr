#!/usr/bin/env bash
set -euo pipefail

# CPU-only fallback. It reuses the three already scheduled n=4 oracle
# rollouts only if the audited seed-11 source misses its coverage gate.
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

for seed in 22 33; do
  while ! test -f "$scale/candidate_seed${seed}.completed"; do
    check_suite
    sleep 60
  done
  test -s "$scale/train_audit_seed${seed}.json"
done
"$base/activevln_train_env/bin/python" \
  "$root/tools/preflight_group_future_advantage_pool.py" \
  --run 11 "$root/verl_checkpoints/oracle_turnwise_exact512_128_seed11/rollout.jsonl" \
    "$scale/train_audit_seed11.json" \
  --run 22 "$root/verl_checkpoints/oracle_turnwise_exact512_128_seed22/rollout.jsonl" \
    "$scale/train_audit_seed22.json" \
  --run 33 "$root/verl_checkpoints/oracle_turnwise_exact512_128_seed33/rollout.jsonl" \
    "$scale/train_audit_seed33.json" \
  --dataset "$dataset" --scene-split "$split" \
  --output "$run/report.json" >"$run/analysis.log" 2>&1
test -s "$run/report.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
