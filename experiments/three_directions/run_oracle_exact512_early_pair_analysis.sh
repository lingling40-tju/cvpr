#!/usr/bin/env bash
set -euo pipefail

# CPU-only paired audit after each candidate's complete 1,839-item eval.
# The final three-seed result remains in run_oracle_exact512_scale_after_full.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
result="$root/runlogs/oracle_exact512_full1839"
eval_watcher="$root/runlogs/oracle_candidate_eval_overlap"
run="$root/runlogs/oracle_exact512_early_pair_analysis"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'early paired audit already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

for seed in 11 22 33; do
  candidate="oracle_turnwise_exact512_128_seed${seed}"
  baseline="oracle_exact512_control_128_seed${seed}"
  if test -f "$run/seed${seed}.completed"; then continue; fi
  while ! test -f "$result/$candidate.completed"; do
    test ! -f "$result/$candidate.failed" || {
      echo "candidate eval failed: seed $seed" >&2; exit 1;
    }
    test ! -f "$eval_watcher/failed" || {
      echo 'candidate eval watcher failed' >&2; exit 1;
    }
    test -s "$eval_watcher/launcher.pid"
    pid=$(cat "$eval_watcher/launcher.pid")
    ps -o args= -p "$pid" | grep -F \
      'run_oracle_candidate_eval_overlap.sh' >/dev/null || {
        echo 'candidate eval watcher vanished' >&2; exit 1;
      }
    sleep 60
  done
  test -f "$result/$baseline.completed"
  "$base/activevln_server_env/bin/python" \
    "$root/tools/check_exact512_eval_progress.py" --root "$result" \
    --labels "$candidate" "$baseline" \
    >"$run/seed${seed}.coverage.json"
  PYTHONPATH="$control/tools${PYTHONPATH:+:$PYTHONPATH}" \
    "$base/activevln_server_env/bin/python" \
    "$root/tools/analyze_oracle_exact512_one_seed.py" \
    --root "$result" --screen-manifest "$root/tools/process_val256_manifest.json" \
    --seed "$seed" --output "$run/seed${seed}.json" \
    >"$run/seed${seed}.analysis.log" 2>&1
  "$base/activevln_server_env/bin/python" \
    "$control/tools/export_matched_episodes.py" \
    --root "$result" --candidate "$candidate" --control "$baseline" \
    --expected-count 1839 --output "$run/seed${seed}.episodes.jsonl" \
    >"$run/seed${seed}.export.log" 2>&1
  test -s "$run/seed${seed}.json"
  test -s "$run/seed${seed}.episodes.jsonl"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
