#!/usr/bin/env bash
set -euo pipefail

# Continue the already-running seed11_control evaluation on GPU1/port8004,
# while evaluating event checkpoints on GPU3/port8005. Each lane uses four
# Habitat workers on GPU2 and the same 1,839-episode manifest.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
log="$root/runlogs/eventtrace_full_val_unseen"
manifest="$log/manifest.json"
mkdir -p "$log"
test -f "$log/sft.completed"

validate() {
  "$base/activevln_server_env/bin/python" "$root/tools/validate_full_label.py" \
    "$1" "$log" "$manifest" 4 >"$log/$1.validated.json"
}

evaluate() {
  local label=$1 gpu=$2 port=$3 model
  if [ -f "$log/$label.completed" ]; then
    validate "$label"
    return
  fi
  model="$root/verl_checkpoints/eventtrace_r2r64_${label}/global_step_64/actor/huggingface"
  test -f "$model/config.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$log/$label.started"
  rm -f "$log/$label.failed"
  if EVENTTRACE_INFERENCE_GPU="$gpu" EVENTTRACE_INFERENCE_PORT="$port" \
      EVENTTRACE_SIM_GPU=2 EVENTTRACE_SHARDS=4 \
      bash "$root/tools/run_full_val_unseen.sh" "$label" "$model" \
      >"$log/$label.run.log" 2>&1; then
    validate "$label"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$log/$label.completed"
  else
    local status=$?
    rm -f "$log/$label.completed"
    printf '%s\n' "$status" >"$log/$label.failed"
    return "$status"
  fi
}

control_lane() {
  # The first control checkpoint was launched by the original scheduler.
  # Its four summaries and a closed port show that it has fully finished.
  while [ ! -f "$log/seed11_control.completed" ]; do
    if [ -f "$log/seed11_control.failed" ]; then return 1; fi
    if [ -f "$log/seed11_control/shard_00/summary.json" ] && \
       [ -f "$log/seed11_control/shard_01/summary.json" ] && \
       [ -f "$log/seed11_control/shard_02/summary.json" ] && \
       [ -f "$log/seed11_control/shard_03/summary.json" ] && \
       ! curl -fsS --max-time 2 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
      validate seed11_control
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$log/seed11_control.completed"
      break
    fi
    sleep 15
  done
  evaluate seed22_control 1 8004
  evaluate seed33_control 1 8004
}

event_lane() {
  evaluate seed11_event 3 8005
  evaluate seed22_event 3 8005
  evaluate seed33_event 3 8005
}

control_lane &
control_pid=$!
event_lane &
event_pid=$!
status=0
wait "$control_pid" || status=1
wait "$event_pid" || status=1
if [ "$status" -ne 0 ]; then exit 1; fi
"$base/activevln_server_env/bin/python" "$root/tools/analyze_full_val.py" \
  >"$log/analysis.log" 2>&1
