#!/usr/bin/env bash
set -euo pipefail
requested_arm=${1:?pass control or candidate}
case "$requested_arm" in control|candidate) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/positive_pilot"
result="$root/runlogs/positive_development256"
# The existing suite calls control, then candidate. Its first call runs the
# fixed pair concurrently; the second returns after both arm validators.
test -f "$state/training.completed"
test ! -f "$state/suite.failed"
"$base/activevln_server_env/bin/python" - "$state" <<'PY'
from pathlib import Path
import json, sys
state = Path(sys.argv[1])
for arm in ("control", "candidate"):
    a = json.loads((state / f"{arm}_train_audit.json").read_text())
    assert a["arm"] == arm and a["expected_steps"] == a["observed_steps"] == 64
    assert a["positive_advantage_steps"] > 0 and a["nonzero_actor_gradient_steps"] > 0
    assert a["kl_loss_metric_present_and_finite"] is True
    assert a["max_terminal_score"] <= 20.001
    if arm == "candidate": assert a["min_advantage"] >= -0.001
PY
mkdir -p "$result"
exec 9>"$state/development_pair.lock"
flock 9
both_complete=1
for arm in control candidate; do
  label="positive_trajectory_${arm}_64step_seed11"
  test -f "$root/runlogs/$label/completed"
  test -f "$root/verl_checkpoints/$label/global_step_64/actor/huggingface/config.json"
  if ! test -f "$result/$label.completed" || test -f "$result/$label.failed"; then both_complete=0; fi
done
if test "$both_complete" -eq 1; then exit 0; fi
for _ in $(seq 1 180); do
  busy=0
  for gpu in 0 1 2; do
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if test "$memory" -ge 10000; then busy=1; fi
  done
  if test "$busy" -eq 0; then break; fi
  sleep 10
done
test "$busy" -eq 0
rm -f "$state/development_pair.failed" "$state/development_pair.completed"
pids=()
cleanup() {
  status=$?
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/development_pair.failed"; fi
}
trap cleanup EXIT
for arm in control candidate; do
  bash "$root/tools/run_positive_development_model.sh" "$arm" \
    >"$state/${arm}_parallel_eval.launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
pids=()
test "$status" -eq 0
for arm in control candidate; do
  label="positive_trajectory_${arm}_64step_seed11"
  test -f "$result/$label.completed"
  test ! -f "$result/$label.failed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/development_pair.completed"
