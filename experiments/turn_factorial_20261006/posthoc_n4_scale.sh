#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_norm_terminal_rloo_20261006"
n8="$base/ActiveVLN_norm_terminal_rloo_n8_20261006/runlogs/posthoc_n8_pilot"
state="$root/runlogs/posthoc_n4_scale"
mkdir -p "$state"
exec 9>"$state/watcher.lock"
flock -n 9 || { echo 'posthoc n4 scale watcher already active' >&2; exit 2; }
if test -f "$state/watcher.completed"; then exit 0; fi
rm -f "$state/watcher.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/watcher.failed"; fi; }
trap on_exit EXIT
while ! test -f "$n8/suite.completed"; do
  test ! -f "$n8/suite.failed" || { echo 'n8 pilot failed; diagnose and release GPUs before n4 scale' >&2; exit 1; }
  test -s "$n8/watcher.launcher.pid"
  pid=$(cat "$n8/watcher.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'n8 watcher disappeared' >&2; exit 1; }
  sleep 60
done
test ! -f "$n8/suite.failed"
for attempt in $(seq 1 360); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 60
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000
! curl -fsS --max-time 2 http://127.0.0.1:5059/health >/dev/null 2>&1
bash "$root/tools/start_service.sh" >"$state/service_start.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/posthoc_n4_train.sh" "$seed" >"$state/seed${seed}.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" "$root/tools/audit_training_gradients.py" \
    "$root/runlogs/norm_terminal_posthoc512_128_seed${seed}/train.log" 128 \
    "$state/seed${seed}_train_audit.json" >"$state/seed${seed}_audit.log"
done
pidfile="$root/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5059' >/dev/null
kill "$pid"
for attempt in $(seq 1 90); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/scale_train.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
bash "$root/tools/posthoc_n4_full_eval.sh" >"$state/full_eval.launcher.log" 2>&1
