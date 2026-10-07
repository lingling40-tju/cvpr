#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
pilot="$root/runlogs/positive_pilot"
state="$root/runlogs/positive_scale"
mkdir -p "$state"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'positive scale already active' >&2; exit 2; }
if test -f "$state/suite.completed" || test -f "$state/scale.skipped"; then exit 0; fi
test ! -f "$state/suite.failed"
service_owned=0
eval_pids=()
stop_service() {
  if test "$service_owned" -ne 1; then return; fi
  if test -s "$root/runlogs/service/server.pid"; then
    pid=$(cat "$root/runlogs/service/server.pid")
    if test "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$root" && \
        ps -o args= -p "$pid" 2>/dev/null | grep -F 'server.port=5085' >/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  fi
  for _ in $(seq 1 90); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5085/health >/dev/null 2>&1; then break; fi
    sleep 2
  done
  service_owned=0
}
cleanup() {
  status=$?
  for pid in "${eval_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  stop_service
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi
}
trap cleanup EXIT
while ! test -f "$pilot/suite.completed"; do
  test ! -f "$pilot/suite.failed"
  test -s "$pilot/watcher.launcher.pid"
  kill -0 "$(cat "$pilot/watcher.launcher.pid")"
  sleep 30
done
# Acquire the pilot's actual process lock before touching its released GPUs.
exec 8>"$pilot/suite.lock"
flock 8
test ! -f "$pilot/suite.failed"
"$base/activevln_server_env/bin/python" "$root/tools/verify_positive_compact.py" \
  --manifest "$root/prepared_data/development256.json" \
  --compact "$pilot/development256_paired.jsonl" --report "$pilot/development256_pair.json" \
  --validators "$root/runlogs/positive_development256" --gate "$pilot/frozen_gate.json" \
  --output "$state/development_independent_recount.json" >"$state/development_recount.log"
flock -u 8
exec 8>&-
decision=$("$base/activevln_server_env/bin/python" - "$pilot/frozen_gate.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
assert g["schema"] == "positive_trajectory_frozen_development_gate_v1"
assert g["requires_each_metric_points_at_least"] == 2.0
assert g["pass"] == (g["paired_sr_points"] >= 2 and g["paired_spl_points"] >= 2)
assert g["reserved_screen_opened"] is False
print("pass" if g["pass"] else "skip")
PY
)
if test "$decision" = skip; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/scale.skipped"
  exit 0
fi
wait_gpus() {
  for _ in $(seq 1 360); do
    busy=0
    for gpu in 0 1 2; do
      memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
      if test "$memory" -ge 10000; then busy=1; fi
    done
    if test "$busy" -eq 0; then return; fi
    sleep 10
  done
  echo 'GPUs remain occupied; refusing to start scale' >&2
  return 1
}
if ! test -f "$state/training.completed"; then
  wait_gpus
  service_owned=1
  bash "$root/tools/start_positive_service.sh" >"$state/service_start.log" 2>&1
  for seed in 11 22 33; do
    for arm in control candidate; do
      bash "$root/tools/run_positive_scale_train.sh" "$arm" "$seed" \
        >"$state/${arm}_seed${seed}_train.launcher.log" 2>&1
      "$base/activevln_train_env/bin/python" "$root/tools/audit_positive_scale_train.py" \
        --log "$root/runlogs/positive_trajectory_${arm}_128step_seed${seed}/train.log" \
        --steps 128 --arm "$arm" --output "$state/${arm}_seed${seed}_train_audit.json" \
        >"$state/${arm}_seed${seed}_audit.log"
    done
  done
  stop_service
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"
fi
if ! test -f "$state/reserved.opened"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/reserved.opened"
fi
for seed in 11 22 33; do
  wait_gpus
  bash "$root/tools/run_positive_reserved_eval.sh" control "$seed" \
    >"$state/control_seed${seed}_eval.launcher.log" 2>&1 &
  control_pid=$!
  bash "$root/tools/run_positive_reserved_eval.sh" candidate "$seed" \
    >"$state/candidate_seed${seed}_eval.launcher.log" 2>&1 &
  candidate_pid=$!
  eval_pids=("$control_pid" "$candidate_pid")
  status=0
  wait "$control_pid" || status=1
  wait "$candidate_pid" || status=1
  eval_pids=()
  test "$status" -eq 0
  "$base/activevln_server_env/bin/python" "$root/tools/analyze_train_scene_pair.py" \
    --root "$root/runlogs/positive_reserved256" --manifest "$root/prepared_data/reserved256.json" \
    --role reserved --control "positive_trajectory_control_128step_seed${seed}" \
    --candidate "positive_trajectory_candidate_128step_seed${seed}" \
    --compact "$state/seed${seed}_reserved_paired.jsonl" \
    --output "$state/seed${seed}_reserved_pair.json" >"$state/seed${seed}_analyze.log"
  "$base/activevln_server_env/bin/python" "$root/tools/verify_positive_compact.py" \
    --manifest "$root/prepared_data/reserved256.json" \
    --compact "$state/seed${seed}_reserved_paired.jsonl" \
    --report "$state/seed${seed}_reserved_pair.json" \
    --validators "$root/runlogs/positive_reserved256" \
    --output "$state/seed${seed}_independent_recount.json" >"$state/seed${seed}_recount.log"
done
"$base/activevln_server_env/bin/python" "$root/tools/analyze_positive_scale.py" \
  --root "$state" --manifest "$root/prepared_data/reserved256.json" \
  --output "$state/three_seed_reserved_report.json" >"$state/three_seed_analyze.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
