#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_future_advantage_sparse_replay.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_run="$root/runlogs/future_advantage_sparse_manifest"
report="$root/runlogs/future_advantage_pooled_preflight/report.json"
manifest="$source_run/replay_manifest.json"
labels="$source_run/pair_labels.json"
scratch="$root/runlogs/future_advantage_sparse_replay"
run="$scratch/$mode"
python="$base/activevln_server_env/bin/python"
gpu=${VLN_SPARSE_GPU:-1}
shards=${VLN_SPARSE_SHARDS:-1}
[[ "$gpu" =~ ^[0-3]$ && "$shards" =~ ^[1-4]$ ]]
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "sparse $mode replay already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$source_run/completed" && test ! -f "$source_run/skipped_coverage_short"
test -s "$report" && test -s "$manifest" && test -s "$labels"
export PYTHONPATH="$root/tools:$root:$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"

# The full-val evaluator uses the same per-GPU lock before launching vLLM.
# Holding it across replay prevents an early train-data replay from racing
# the next evaluation model on GPU 1.
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu${gpu}.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8

check_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$((gpu + 1))p" | tr -d ' ')
  test -n "$used" && test "$used" -lt 5000 || {
    echo "GPU $gpu occupied ($used MiB); defer sparse replay" >&2
    return 1
  }
}

if test "$mode" = smoke; then
  read -r last_seed rid < <("$python" - "$manifest" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='future_advantage_sparse_replay_manifest_v1'
assert x['group_size']==4
plans=[p for p in x['selected']['fit'] if p['anchor_turns']==[3,6]]
assert plans
print(max(x['seeds']), plans[0]['record_id'])
PY
)
  test -f "$root/runlogs/oracle_exact512_full1839/oracle_turnwise_exact512_128_seed${last_seed}.completed"
  check_gpu
  baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$((gpu + 1))p" | tr -d ' ')
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
    sed -n "$((gpu + 1))p" | tr -d ' ')
  start_epoch=$(date +%s)
  (while :; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
      sed -n "$((gpu + 1))p" | tr -d ' '
    sleep 1
  done) >"$run/gpu_used_mib.log" &
  monitor=$!
  if ! "$python" "$root/tools/collect_future_advantage_sparse_frames.py" \
    --manifest "$manifest" --report "$report" --root "$root" \
    --part fit --output-root "$run/rgb" --gpu "$gpu" \
    --record-id "$rid" >"$run/collect.log" 2>&1; then
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    exit 1
  fi
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  end_epoch=$(date +%s)
  "$python" - "$run" "$baseline" "$total" "$start_epoch" "$end_epoch" <<'PY' >"$run/resource.log"
import json,pathlib,sys
run=pathlib.Path(sys.argv[1])
baseline,total,start,end=map(int,sys.argv[2:])
samples=[int(line) for line in (run/'gpu_used_mib.log').read_text().splitlines()
         if line.strip().isdigit()]
assert samples and 0<=baseline<=total and end>=start
peak=max(baseline,*samples)
result={'schema':'future_advantage_sparse_smoke_resource_v1',
        'wall_seconds':max(1,end-start),'baseline_used_mib':baseline,
        'peak_used_mib':peak,'incremental_peak_mib':peak-baseline,
        'gpu_total_mib':total,'samples':len(samples)}
(run/'resource.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
  "$python" - "$run" "$manifest" "$rid" <<'PY' >"$run/smoke_audit.log"
import hashlib,json,math,pathlib,sys
from PIL import Image
run,manifest,rid=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
part=run/'rgb/fit'
sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
record=json.loads((part/'records'/f'{rid}.json').read_text())
audit=json.loads((part/'audits'/f'{rid}.json').read_text())
summary=json.loads((part/'summary.json').read_text())
assert record['manifest_sha256']==audit['manifest_sha256']==summary['manifest_sha256']==sha
assert record['record_id']==audit['record_id']==rid
assert set(record['input'])=={'instruction','images','action_history_by_anchor'}
assert set(record['input']['images'])=={'0','3','6'}
assert set(record['input']['action_history_by_anchor'])=={'3','6'}
assert summary['requested']==summary['completed']==1 and not summary['errors']
assert math.isfinite(audit['terminal_drift_m']) and abs(audit['terminal_drift_m'])<=.25
for relative in record['input']['images'].values():
    with Image.open(part/relative) as frame: frame.verify()
result={'schema':'future_advantage_sparse_smoke_audit_v1',
        'record_id':rid,'manifest_sha256':sha,'images':3,
        'terminal_drift_m':audit['terminal_drift_m'],
        'navigation_result':False}
(run/'smoke_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
  test -s "$run/smoke_audit.json"
  test -s "$run/resource.json"
else
  test -f "$scratch/smoke/completed" && test -s "$scratch/smoke/smoke_audit.json"
  "$python" - "$scratch/smoke/smoke_audit.json" "$manifest" <<'PY'
import hashlib,json,sys
audit=json.load(open(sys.argv[1]))
assert audit['manifest_sha256']==hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
PY
  # An explicitly early replay may use an idle GPU while the final seed
  # trains. The shared GPU lock above makes any evaluator wait for replay.
  if test "${VLN_SPARSE_EARLY_FULL:-0}" = 1; then
    test "$gpu" -eq 1
    test -f "$root/runlogs/oracle_exact512_full1839/oracle_turnwise_exact512_128_seed22.completed"
  else
    test -f "$root/runlogs/oracle_candidate_eval_overlap/completed"
  fi
  for part in fit development; do
    check_gpu
    pids=()
    for ((shard=0; shard<shards; shard++)); do
      "$python" "$root/tools/collect_future_advantage_sparse_frames.py" \
        --manifest "$manifest" --report "$report" --root "$root" \
        --part "$part" --output-root "$run/rgb" --gpu "$gpu" \
        --shards "$shards" --shard "$shard" \
        >"$run/$part.shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    test "$status" -eq 0
  done
  "$python" "$root/tools/verify_future_advantage_sparse_replay.py" \
    --manifest "$manifest" --labels "$labels" --report "$report" \
    --replay-root "$run/rgb" --output "$run/verification.json" \
    >"$run/verification.log" 2>&1
  test -s "$run/verification.json"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
