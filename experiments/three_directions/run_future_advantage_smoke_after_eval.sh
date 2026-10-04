#!/usr/bin/env bash
set -euo pipefail

# Launch one real, label-only sparse Habitat replay as soon as the GPU-1
# full-evaluation lane ends. The full 1,795-record replay needs a separate
# shard decision after this smoke has been timed and audited.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
eval_run="$root/runlogs/oracle_candidate_eval_overlap"
source_run="$root/runlogs/future_advantage_sparse_manifest"
pool="$root/runlogs/future_advantage_pooled_preflight"
run="$root/runlogs/future_advantage_sparse_replay/smoke_watcher"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'sparse smoke watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
echo "$$" >"$run/launcher.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
test -f "$source_run/completed" && test -s "$pool/report.json"
"$base/activevln_train_env/bin/python" - "$pool/report.json" \
  "$source_run/replay_manifest.json" <<'PY'
import json,sys
report,manifest=(json.load(open(path)) for path in sys.argv[1:])
assert report['seeds']==manifest['seeds']==[11,22]
assert report['group_size']==manifest['group_size']==4
assert report['enough_coverage_for_fit_preparation']
assert all(report['coverage_checks'].values())
PY

while ! test -f "$eval_run/completed"; do
  test ! -f "$eval_run/failed" || {
    echo 'candidate full-evaluation lane failed' >&2; exit 1;
  }
  test -s "$eval_run/launcher.pid"
  pid=$(cat "$eval_run/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_candidate_eval_overlap.sh' >/dev/null || {
      echo 'candidate full-evaluation watcher vanished' >&2; exit 1;
    }
  sleep 60
done
for seed in 11 22 33; do
  test -f "$root/runlogs/oracle_exact512_full1839/oracle_turnwise_exact512_128_seed${seed}.completed"
done
# Give vLLM time to release GPU 1 after the evaluator's completion marker.
for _ in {1..30}; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n 2p | tr -d ' ')
  if test -n "$used" && test "$used" -lt 5000; then break; fi
  sleep 30
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 still occupied ($used MiB)" >&2; exit 1;
}
bash "$root/tools/run_future_advantage_sparse_replay.sh" smoke \
  >"$run/smoke.log" 2>&1
test -f "$root/runlogs/future_advantage_sparse_replay/smoke/completed"
test -s "$root/runlogs/future_advantage_sparse_replay/smoke/smoke_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
