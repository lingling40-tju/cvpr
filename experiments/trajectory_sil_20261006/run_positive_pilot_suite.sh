#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
n8="$base/ActiveVLN_norm_terminal_rloo_n8_20261006/runlogs/posthoc_n8_pilot"
n4="$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/posthoc_n4_full1839"
state="$root/runlogs/positive_pilot"
result="$root/runlogs/positive_development256"
mkdir -p "$state" "$result"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'positive-trajectory pilot already active' >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
test -f "$n8/suite.completed" && test ! -f "$n8/suite.failed"
test -f "$n4/suite.completed" && test ! -f "$n4/suite.failed"
rm -f "$state/suite.failed"

stop_own_service() {
  pidfile="$root/runlogs/service/server.pid"
  if test -s "$pidfile"; then
    pid=$(cat "$pidfile")
    if ps -o args= -p "$pid" 2>/dev/null | grep -F 'server.port=5085' >/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  fi
  for _ in $(seq 1 90); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5085/health >/dev/null 2>&1; then break; fi
    sleep 2
  done
}
on_exit() {
  status=$?
  stop_own_service
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi
}
trap on_exit EXIT

for _ in $(seq 1 180); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000
bash "$root/tools/start_positive_service.sh" >"$state/service_start.log" 2>&1
for arm in control candidate; do
  bash "$root/tools/run_positive_train.sh" "$arm" 2 >"$state/${arm}_smoke.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" "$root/tools/audit_positive_train.py" \
    --log "$root/runlogs/positive_trajectory_${arm}_2step_seed11/train.log" \
    --steps 2 --arm "$arm" --output "$state/${arm}_smoke_audit.json" \
    >"$state/${arm}_smoke_audit.log"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/smoke.completed"
for arm in control candidate; do
  bash "$root/tools/run_positive_train.sh" "$arm" 64 >"$state/${arm}_train.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" "$root/tools/audit_positive_train.py" \
    --log "$root/runlogs/positive_trajectory_${arm}_64step_seed11/train.log" \
    --steps 64 --arm "$arm" --output "$state/${arm}_train_audit.json" \
    >"$state/${arm}_train_audit.log"
done
stop_own_service
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"

for arm in control candidate; do
  for _ in $(seq 1 180); do
    mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
    if test "$mem0" -lt 10000 && test "$mem2" -lt 10000; then break; fi
    sleep 10
  done
  test "$mem0" -lt 10000 && test "$mem2" -lt 10000
  bash "$root/tools/run_positive_development_eval.sh" "$arm" \
    >"$state/${arm}_eval.launcher.log" 2>&1
done
"$base/activevln_server_env/bin/python" "$root/tools/analyze_train_scene_pair.py" \
  --root "$result" --manifest "$root/prepared_data/development256.json" \
  --role development \
  --control positive_trajectory_control_64step_seed11 \
  --candidate positive_trajectory_candidate_64step_seed11 \
  --compact "$state/development256_paired.jsonl" \
  --output "$state/development256_pair.json" >"$state/analyze_pair.log"
"$base/activevln_server_env/bin/python" - "$state/development256_pair.json" "$state/frozen_gate.json" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]))
gate={"schema":"positive_trajectory_frozen_development_gate_v1",
      "paired_sr_points":report["paired_sr_points"],
      "paired_spl_points":report["paired_spl_points"],
      "requires_each_metric_points_at_least":2.0,
      "pass":report["paired_sr_points"]>=2.0 and report["paired_spl_points"]>=2.0,
      "reserved_screen_opened":False}
with open(sys.argv[2],"w") as handle: json.dump(gate,handle,indent=2)
print(json.dumps(gate))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
