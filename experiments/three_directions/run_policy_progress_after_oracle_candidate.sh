#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
oracle="$base/ActiveVLN_turnwise_oracle_20261004"
scratch="$base/policy_progress_lora_20261004"
run="$scratch/runlogs/after_oracle_candidate"
candidate=oracle_turnwise_64_seed11_full_recheck
mkdir -p "$run"
exec 9>"$run/watcher.lock"
flock -n 9 || { echo 'progress LoRA watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
while ! test -f "$oracle/runlogs/oracle_full1839/$candidate.completed"; do
  if test -f "$oracle/runlogs/oracle_full1839/$candidate.failed" || \
     test -f "$oracle/runlogs/oracle_full_recheck/failed"; then
    echo 'oracle candidate evaluation failed before GPU3 release' >&2
    exit 1
  fi
  sleep 30
done
while true; do
  used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n '4p' | tr -d ' ')
  if test -n "$used3" && test "$used3" -lt 8000 && \
     ! curl -fsS --max-time 2 'http://127.0.0.1:8101/v1/models' >/dev/null 2>&1; then
    break
  fi
  sleep 15
done
bash "$scratch/run_policy_progress_lora.sh" smoke
bash "$scratch/run_policy_progress_lora.sh" full
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
