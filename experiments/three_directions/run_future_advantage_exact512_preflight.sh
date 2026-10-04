#!/usr/bin/env bash
set -euo pipefail

# Do a CPU-only label-coverage inventory as soon as the first audited
# exact512 n=4 oracle arm completes; later seeds can keep training.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
scale="$root/runlogs/oracle_exact512_scale"
candidate="$root/verl_checkpoints/oracle_turnwise_exact512_128_seed11"
dataset="$root/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
split="$base/ActiveVLN_three_directions_20261002/runlogs/ordinal_progress/stop_history_lora_scene_split.json"
run="$root/runlogs/future_advantage_exact512_preflight"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'future-advantage preflight already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
while ! test -f "$scale/candidate_seed11.completed"; do
  test ! -f "$scale/suite.failed" || { echo 'n=4 suite failed' >&2; exit 1; }
  test -s "$scale/launcher.pid"
  pid=$(cat "$scale/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_exact512_scale_after_full.sh' >/dev/null || {
      echo 'n=4 suite launcher vanished' >&2; exit 1;
    }
  sleep 60
done
test -s "$scale/train_audit_seed11.json"
test -s "$candidate/rollout.jsonl"
sha256sum "$root/tools/preflight_group_future_advantage.py" \
  "$root/tools/preflight_group_future_advantage_exact512.py" \
  "$scale/train_audit_seed11.json" "$candidate/rollout.jsonl" \
  "$dataset" "$split" >"$run/source.sha256"
"$base/activevln_train_env/bin/python" \
  "$root/tools/preflight_group_future_advantage_exact512.py" \
  --rollout "$candidate/rollout.jsonl" \
  --dataset "$dataset" --scene-split "$split" \
  --train-audit "$scale/train_audit_seed11.json" \
  --output "$run/report.json" >"$run/analysis.log" 2>&1
test -s "$run/report.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
