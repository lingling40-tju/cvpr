#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_norm_terminal_rloo_20261006"
gae="$base/ActiveVLN_turn_gae_20261006/runlogs/conditional_chain"
chain="$root/runlogs/conditional_chain"
state="$root/runlogs/normalized_scale"
mkdir -p "$state"
exec 9>"$state/watcher.lock"
flock -n 9 || { echo 'normalized scale watcher already active' >&2; exit 2; }
if test -f "$state/watcher.completed"; then exit 0; fi
rm -f "$state/watcher.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/watcher.failed"; fi; }
trap on_exit EXIT

while ! test -f "$gae/skipped_for_scale.completed" && ! test -f "$gae/normalized_failed_gate.completed"; do
  test ! -f "$chain/chain.failed" && test ! -f "$gae/watcher.failed" || {
    echo 'conditional decision failed; scaling requires recovery' >&2; exit 1;
  }
  test -s "$chain/chain.launcher.pid" && test -s "$gae/watcher.launcher.pid"
  for pidfile in "$chain/chain.launcher.pid" "$gae/watcher.launcher.pid"; do
    pid=$(cat "$pidfile")
    kill -0 "$pid" 2>/dev/null || { echo "watcher disappeared: $pidfile" >&2; exit 1; }
  done
  sleep 60
done
if test -f "$gae/normalized_failed_gate.completed"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/skipped_after_failed_pilot.completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
  exit 0
fi
while ! test -f "$gae/watcher.completed"; do
  test ! -f "$gae/watcher.failed" || { echo 'GAE conditional gate watcher failed' >&2; exit 1; }
  pid=$(cat "$gae/watcher.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'GAE conditional gate watcher disappeared' >&2; exit 1; }
  sleep 2
done
test -f "$chain/chain.completed"
test ! -f "$gae/watcher.failed"
test -f "$gae/skipped_for_scale.completed"

for port in 5057 5058 5059 5060; do
  ! curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1
done
for attempt in $(seq 1 60); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000
bash "$root/tools/start_service.sh"
for seed in 11 22 33; do
  bash "$root/tools/normalized_scale_train.sh" "$seed"
  "$base/activevln_train_env/bin/python" \
    "$base/ActiveVLN_turn_rloo_terminal_20261006/tools/audit_training_gradients.py" \
    "$root/runlogs/norm_terminal_rloo_exact512_128_seed${seed}/train.log" 128 \
    "$state/seed${seed}_train_audit.json" >"$state/seed${seed}_audit.log"
done
pidfile="$root/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5059' >/dev/null
kill "$pid"
for attempt in $(seq 1 60); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/scale_train.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
