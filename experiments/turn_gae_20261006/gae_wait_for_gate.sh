#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
normalized="$base/ActiveVLN_norm_terminal_rloo_20261006"
suite="$normalized/runlogs/norm_terminal_val_seen256"
prior="$normalized/runlogs/conditional_chain"
state="$root/runlogs/conditional_chain"
mkdir -p "$state"
exec 9>"$state/watcher.lock"
flock -n 9 || { echo 'GAE conditional watcher already active' >&2; exit 2; }
if test -f "$state/watcher.completed"; then exit 0; fi
rm -f "$state/watcher.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/watcher.failed"; fi; }
trap on_exit EXIT
while ! test -f "$prior/chain.completed"; do
  test ! -f "$prior/chain.failed" && test ! -f "$suite/suite.failed" || {
    echo 'normalized suite failed; GAE cannot make a conditional decision' >&2; exit 1;
  }
  test -s "$prior/chain.launcher.pid"
  pid=$(cat "$prior/chain.launcher.pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    test -f "$prior/chain.completed" || { echo 'normalized watcher disappeared' >&2; exit 1; }
  fi
  sleep 60
done
test -f "$suite/suite.completed"
test ! -f "$suite/suite.failed"
test "$(sha256sum "$root/tools/verify_normalized_terminal.py" | awk '{print $1}')" = 1b27e08ca2728f9afb64894a038c2c4f35214b99e95b054f54b13307e43b7dff
"$base/activevln_server_env/bin/python" "$root/tools/verify_normalized_terminal.py" \
  "$suite" "$suite/manifest.json" \
  --compact "$state/normalized_compact.json" \
  --output "$state/normalized_independent_recount.json" \
  >"$state/normalized_recount.log"
"$base/activevln_server_env/bin/python" "$root/tools/gae_gate_from_normalized.py" \
  "$suite" "$state/normalized_independent_recount.json" "$state" \
  >"$state/gate.log"
if test -f "$state/skipped_for_scale.completed"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
  exit 0
fi
test -f "$state/normalized_failed_gate.completed"
for port in 5057 5058 5059 5075; do
  ! curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1
done
for attempt in $(seq 1 60); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem3=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000
bash "$root/tools/gae_start_service.sh"
bash "$root/tools/gae_run_train.sh" 2
"$base/activevln_train_env/bin/python" "$root/tools/audit_gae_gradients.py" \
  "$root/runlogs/turn_gae_2step_seed11/train.log" 2 "$state/gae_2_train_audit.json" \
  --require-each-nonzero >"$state/gae_2_audit.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/smoke.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
