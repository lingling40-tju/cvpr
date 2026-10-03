#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
prior="$base/ActiveVLN_qwen_group_rank_20261004"
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
recheck="$prior/runlogs/qwen_full_recheck"
run="$root/runlogs/oracle_after_recheck"
mkdir -p "$run"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'oracle smoke watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
service_started=0
stop_service() {
  local pid_file="$root/runlogs/oracle_turnwise_service/habitat.pid" pid
  if test "$service_started" -ne 1 || ! test -s "$pid_file"; then return 0; fi
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F 'server.port=5035' >/dev/null || {
      echo "refusing to stop unmatched PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5035/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo 'oracle Habitat port 5035 did not stop' >&2
  return 1
}
on_exit() {
  local status=$?
  stop_service || status=1
  if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$recheck/suite.completed"; do
  if test -f "$recheck/suite.failed"; then
    echo 'full recheck failed; refusing concurrent oracle smoke' >&2
    exit 1
  fi
  if test -s "$recheck/launcher.pid" && \
      ! kill -0 "$(cat "$recheck/launcher.pid")" 2>/dev/null; then
    echo 'full recheck launcher disappeared without terminal marker' >&2
    exit 1
  fi
  sleep 30
done
test "$(cat "$recheck/confidence_decision.txt")" = recheck
test -s "$recheck/qwen_full_recheck_analysis.json"
test -f "$prior/runlogs/qwen_confident_scale_watcher/suite.completed"

# The two full-evaluation lanes must release their model servers first.
for port in 8081 8082 5035; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 || \
      curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "port $port still in use after full recheck" >&2
    exit 1
  fi
done
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[2]}" -lt 8000 && \
      test "${used[3]}" -lt 8000; then
    break
  fi
  sleep 30
done
test "${used[0]}" -lt 8000 && test "${used[2]}" -lt 8000 && \
  test "${used[3]}" -lt 8000
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/gpus_released"

service_started=1
bash "$root/tools/start_oracle_turnwise_service.sh" \
  >"$run/service_launcher.log" 2>&1
bash "$root/tools/run_oracle_turnwise_train.sh" 2 \
  >"$run/train_2step.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" \
  "$root/tools/audit_oracle_turnwise_train.py" \
  --root "$root" --control-root "$control" --steps 2 \
  --output "$run/train_audit_2step.json" \
  >"$run/train_audit_2step.log" 2>&1
test -f "$root/runlogs/oracle_turnwise_2step_seed11/audited"
test -s "$run/train_audit_2step.json"
stop_service
service_started=0
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
