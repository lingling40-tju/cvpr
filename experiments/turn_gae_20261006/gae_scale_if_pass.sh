#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
result="$root/runlogs/turn_gae_val_seen778"
normal="$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/posthoc_n4_full1839"
normal_scale="$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/posthoc_n4_scale"
state="$root/runlogs/conditional_three_seed_scale"
mkdir -p "$state"
exec 9>"$state/watcher.lock"
flock -n 9 || { echo 'conditional GAE scale watcher already active' >&2; exit 2; }
if test -f "$state/watcher.completed"; then exit 0; fi
rm -f "$state/watcher.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/watcher.failed"; fi; }
trap on_exit EXIT
while ! test -f "$result/suite.verified"; do
  test ! -f "$result/suite.failed" || { echo 'GAE paired evaluation failed' >&2; exit 1; }
  test -s "$result/suite.launcher.pid"
  pid=$(cat "$result/suite.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'GAE paired evaluation watcher disappeared' >&2; exit 1; }
  sleep 60
done
test ! -f "$result/suite.failed"
gate=$(python3 - "$result/verified_gate.json" <<'GATE'
import json,sys
x=json.load(open(sys.argv[1]))
assert isinstance(x['advance_gate_passed'],bool)
sr=float(x['paired_sr_points']); spl=float(x['paired_spl_points'])
assert x['advance_gate_passed']==(sr>=2 and spl>=2)
print('pass' if x['advance_gate_passed'] else 'fail')
GATE
)
if test "$gate" = fail; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/skipped_after_failed_gate.completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
  exit 0
fi
test "$gate" = pass
while ! test -f "$normal/suite.completed"; do
  test ! -f "$normal/suite.failed" && test ! -f "$normal_scale/watcher.failed" || { echo 'preceding normalized suite failed; recover before GAE scale' >&2; exit 1; }
  test -s "$normal_scale/watcher.launcher.pid"
  pid=$(cat "$normal_scale/watcher.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'normalized scale watcher disappeared' >&2; exit 1; }
  sleep 60
done
test ! -f "$normal/suite.failed"
for attempt in $(seq 1 360); do
  busy=0
  for port in 5059 5062 5075 8130 8131 8132 8133 8134; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then busy=1; fi
  done
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem3=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$busy" -eq 0 && test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000; then break; fi
  sleep 60
done
test "$busy" -eq 0 && test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000
bash "$root/tools/gae_start_service.sh" >"$state/service_start.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/gae_scale_train.sh" "$seed" >"$state/seed${seed}.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" "$root/tools/audit_gae_gradients.py" \
    "$root/runlogs/turn_gae_exact512_128_seed${seed}/train.log" 128 \
    "$state/seed${seed}_gradient_audit.json" >"$state/seed${seed}_audit.log"
done
pidfile="$root/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5075' >/dev/null
kill "$pid"
for attempt in $(seq 1 90); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5075/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5075/health >/dev/null 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/scale_train.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
bash "$root/tools/gae_scale_full_eval.sh" >"$state/full_eval.launcher.log" 2>&1
