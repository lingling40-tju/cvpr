#!/usr/bin/env bash
set -euo pipefail

# Wait for both pilot gates. Evaluate every eligible three-seed scale on the
# same fixed 256 and complete 1839 manifests as its seed-matched control.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run_dir="$root/runlogs/optimizer_scale_eval_conditional"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'optimizer evaluation watcher already active' >&2; exit 2; }
if test -f "$run_dir/suite.completed"; then exit 0; fi
rm -f "$run_dir/suite.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

eligible_modes=()
for mode in group4 kl_anchor; do
  gate="$root/runlogs/${mode}_scale_conditional"
  until test -f "$gate/suite.completed"; do
    if test -f "$gate/suite.failed"; then echo "$mode scale gate failed" >&2; exit 1; fi
    pid_file="$root/runlogs/${mode}_scale_conditional_launcher.pid"
    test -s "$pid_file"
    pid=$(cat "$pid_file")
    kill -0 "$pid" 2>/dev/null || { echo "$mode scale watcher PID $pid stopped" >&2; exit 1; }
    sleep 60
  done
  decision=$(cat "$gate/pilot_decision.txt")
  case "$decision" in
    eligible)
      test -f "$gate/training.completed"
      for seed in 11 22 33; do test -f "$gate/train_seed${seed}.completed"; done
      eligible_modes+=("$mode") ;;
    ineligible) test -f "$gate/no_pilot_gain" ;;
    *) echo "invalid $mode pilot decision: $decision" >&2; exit 1 ;;
  esac
done

# These are dedicated, original-reward Habitat services. No trainer can still
# need them after both gate suites have completed.
for entry in group4:5013 kl_anchor:5014; do
  mode=${entry%:*}
  port=${entry#*:}
  pid_file="$root/runlogs/${mode}_service/server.pid"
  if test -s "$pid_file"; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      ps -o args= -p "$pid" | grep -F "server.port=$port" >/dev/null || {
        echo "refusing to stop unmatched service PID $pid" >&2; exit 1;
      }
      kill "$pid"
      for attempt in $(seq 1 30); do
        if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
        sleep 2
      done
    fi
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "dedicated service on $port did not stop" >&2; exit 1
  fi
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/services.stopped"

if [ "${#eligible_modes[@]}" -eq 0 ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/no_eligible_modes"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
  exit 0
fi

eval_mode() {
  local mode=$1 inference_gpu sim_gpu port count result seed label control experiment checkpoint pair
  if [ "$mode" = group4 ]; then
    inference_gpu=3; sim_gpu=2; port=8014
  else
    inference_gpu=1; sim_gpu=0; port=8015
  fi
  for count in 256 1839; do
    if [ "$count" = 256 ]; then
      result="$root/runlogs/three_direction_val256"
      expected_sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
    else
      result="$root/runlogs/three_direction_full_val_unseen"
      expected_sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
    fi
    test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$expected_sha"
    for seed in 11 22 33; do
      label="${mode}_128_seed${seed}"
      control="branch_control128_seed${seed}"
      experiment="three_directions_${mode}_128step"
      if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
      checkpoint="$root/verl_checkpoints/$experiment/global_step_128/actor/huggingface"
      test -f "$root/runlogs/$experiment/completed"
      test -f "$checkpoint/config.json"
      test -f "$result/$control.completed"
      VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_COUNT="$count" VLN_EVAL_PORT="$port" \
        bash tools/run_direction_eval.sh "$label" "$checkpoint" "$inference_gpu" "$sim_gpu" \
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
      --mode "$mode" --root "$result" --count "$count" \
      --output "$result/scale_${mode}_128_analysis.json" \
      >"$run_dir/aggregate_${count}_${mode}.log" 2>&1
    test -s "$result/scale_${mode}_128_analysis.json"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/aggregate_${count}_${mode}.completed"
  done
}

pids=()
for mode in "${eligible_modes[@]}"; do
  eval_mode "$mode" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
