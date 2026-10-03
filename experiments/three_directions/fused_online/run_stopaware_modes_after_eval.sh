#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
run="$root/runlogs/stopaware_modes"
eval_run="$root/runlogs/stopaware_eval256"
result="$root/runlogs/stopaware_val256"
mkdir -p "$run"
exec 9>"$run/modes.lock"
flock -n 9 || { echo 'stop-aware mode analysis already running' >&2; exit 2; }
if test -f "$run/modes.completed"; then exit 0; fi
rm -f "$run/modes.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/modes.failed"; fi
}
trap on_exit EXIT
until test -f "$eval_run/eval.completed"; do
  if test -f "$eval_run/eval.failed" or \
      test -f "$root/runlogs/stopaware_followup/followup.failed"; then
    echo 'paired evaluation failed' >&2; exit 1
  fi
  pid=$(cat "$root/runlogs/stopaware_followup/launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "follow-up watcher PID $pid stopped" >&2; exit 1; }
  sleep 60
done
test -f "$result/stopaware_group4_64_seed11.completed"
test -f "$result/group4_64_seed11_fresh256.completed"
"$base/activevln_server_env/bin/python" "$root/tools/analyze_stopaware_modes.py" \
  --eval-root "$result" \
  --train-rollout "$root/verl_checkpoints/stopaware_group4_64step_seed11/rollout.jsonl" \
  --output "$run/modes.json" >"$run/analysis.log" 2>&1
test -s "$run/modes.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/modes.completed"
