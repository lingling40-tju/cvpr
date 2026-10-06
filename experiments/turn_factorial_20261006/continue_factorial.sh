#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
terminal="$base/ActiveVLN_turn_rloo_terminal_20261006"
dense="$base/ActiveVLN_dense_grpo_20261006"
state="$terminal/runlogs/factorial_chain"
mkdir -p "$state"
exec 9>"$state/chain.lock"
flock -n 9 || { echo "factorial chain already active" >&2; exit 2; }
if test -f "$state/chain.completed"; then exit 0; fi
rm -f "$state/chain.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$state/chain.failed"; fi; }
trap on_exit EXIT
audit="$terminal/tools/audit_training_gradients.py"
wait_for_terminal() {
  local run="$terminal/runlogs/turn_rloo_terminal_64step_seed11"
  local pidfile="$terminal/runlogs/turn_rloo_terminal_64.launcher.pid"
  while ! test -f "$run/completed"; do
    test ! -f "$run/failed" || { echo "terminal training failed" >&2; exit 1; }
    test -s "$pidfile"
    pid=$(cat "$pidfile")
    if ! kill -0 "$pid" 2>/dev/null; then
      test -f "$run/completed" || { echo "terminal launcher vanished" >&2; exit 1; }
    fi
    sleep 60
  done
}
stop_service() {
  local root="$1" port="$2"
  local pidfile="$root/runlogs/service/server.pid"
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    test -s "$pidfile"
    pid=$(cat "$pidfile")
    ps -o args= -p "$pid" | grep -F "server.port=$port" >/dev/null
    kill "$pid"
    for attempt in $(seq 1 60); do
      if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
      sleep 2
    done
    ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1
  fi
}
wait_for_terminal
python3 "$audit" "$terminal/runlogs/turn_rloo_terminal_64step_seed11/train.log" 64 \
  "$state/terminal_64_train_audit.json"
stop_service "$terminal" 5057
bash "$dense/tools/start_service.sh"
bash "$dense/tools/run_train.sh" 2
python3 "$audit" "$dense/runlogs/dense_grpo_2step_seed11/train.log" 2 \
  "$state/dense_2_train_audit.json"
bash "$dense/tools/run_train.sh" 64
python3 "$audit" "$dense/runlogs/dense_grpo_64step_seed11/train.log" 64 \
  "$state/dense_64_train_audit.json"
stop_service "$dense" 5058
bash "$terminal/tools/run_factorial_suite.sh"
test -f "$terminal/runlogs/factorial_val_seen256/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/chain.completed"
