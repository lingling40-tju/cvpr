#!/usr/bin/env bash
set -euo pipefail

# Wait for the real one-record replay, then run verified full replay and the
# fixed development fit. This never launches an online RL reward/policy run.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
scratch="$root/runlogs/future_advantage_sparse_replay"
smoke="$scratch/smoke"
watcher="$scratch/smoke_watcher"
run="$scratch/pipeline"
fit="$root/runlogs/future_advantage_sparse_lora"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'sparse replay/fit pipeline already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
echo "$$" >"$run/launcher.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$watcher/completed"; do
  test ! -f "$watcher/failed" || {
    echo 'real one-record replay failed' >&2; exit 1;
  }
  test -s "$watcher/launcher.pid"
  pid=$(cat "$watcher/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_future_advantage_smoke_after_eval.sh' >/dev/null || {
      echo 'one-record replay watcher vanished' >&2; exit 1;
    }
  sleep 60
done
test -s "$smoke/smoke_audit.json" && test -s "$smoke/resource.json"
test -f "$root/runlogs/oracle_candidate_eval_overlap/completed"
shards=$("$base/activevln_train_env/bin/python" \
  "$root/tools/choose_future_advantage_replay_shards.py" \
  --resource "$smoke/resource.json")
printf 'VLN_SPARSE_SHARDS=%s\n' "$shards" >"$run/resource_decision.txt"
VLN_SPARSE_SHARDS="$shards" VLN_SPARSE_GPU=1 \
  bash "$root/tools/run_future_advantage_sparse_replay.sh" full \
  >"$run/full.log" 2>&1
test -f "$scratch/full/completed" && test -s "$scratch/full/verification.json"
for _ in {1..30}; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n 2p | tr -d ' ')
  if test -n "$used" && test "$used" -lt 5000; then break; fi
  sleep 30
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 still occupied after sparse replay ($used MiB)" >&2; exit 1;
}
bash "$root/tools/run_future_advantage_sparse_fit.sh" \
  >"$run/fit.log" 2>&1
test -f "$fit/completed" && test -s "$fit/development.json"
test -f "$fit/passed_development_gate" || \
  test -f "$fit/failed_development_gate"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
