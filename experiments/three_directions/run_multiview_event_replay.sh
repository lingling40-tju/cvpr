#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_multiview_event_replay.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_dir="$root/runlogs/multiview_event_source"
boundary="$root/runlogs/boundary_occupancy_source/replay_manifest.json"
old_rgb="$root/runlogs/boundary_occupancy_rgb/full/rgb"
scratch="$root/runlogs/multiview_event_rgb"
run="$scratch/$mode"
python="$base/activevln_server_env/bin/python"
gpu=${VLN_EVENT_GPU:-1}
shards=${VLN_EVENT_SHARDS:-4}
[[ "$gpu" =~ ^[0-3]$ && "$shards" =~ ^[1-4]$ ]]
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "event $mode replay already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$root/runlogs/boundary_occupancy_rgb/full/completed"
test -s "$source_dir/capture_manifest.json" && \
  test -s "$source_dir/privileged_pair_labels.json" && \
  test -s "$source_dir/frozen_report.json"
export PYTHONPATH="$root/tools:$root:$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu${gpu}.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8
check_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$((gpu + 1))p" | tr -d ' ')
  test -n "$used" && test "$used" -lt 5000 || {
    echo "GPU $gpu occupied ($used MiB); defer event replay" >&2
    return 1
  }
}
common=(
  --capture-manifest "$source_dir/capture_manifest.json"
  --source-report "$source_dir/frozen_report.json"
  --boundary-manifest "$boundary"
  --old-rgb-root "$old_rgb"
  --root "$root" --output-root "$run/rgb" --gpu "$gpu"
)
verify=(
  --capture-manifest "$source_dir/capture_manifest.json"
  --pair-labels "$source_dir/privileged_pair_labels.json"
  --source-report "$source_dir/frozen_report.json"
  --boundary-manifest "$boundary"
  --old-rgb-root "$old_rgb" --rgb-root "$run/rgb"
  --output "$run/verification.json"
)
if test "$mode" = smoke; then
  check_gpu
  rid=s11_e5986_v2
  "$python" "$root/tools/collect_multiview_event_frames.py" \
    "${common[@]}" --part fit --record-id "$rid" \
    >"$run/collect.log" 2>&1
  "$python" "$root/tools/verify_multiview_event_replay.py" \
    "${verify[@]}" --smoke-record-id "$rid" \
    >"$run/verification.log" 2>&1
else
  test -f "$scratch/smoke/completed" && test -s "$scratch/smoke/verification.json"
  "$python" - "$scratch/smoke/verification.json" "$source_dir/capture_manifest.json" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1]))
assert x['capture_manifest_sha256']==hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
assert x['parts'][0]['records']==1 and x['parts'][0]['new_rendered_images']>=1
PY
  for part in fit development; do
    check_gpu
    pids=()
    for ((shard=0; shard<shards; shard++)); do
      "$python" "$root/tools/collect_multiview_event_frames.py" \
        "${common[@]}" --part "$part" --shards "$shards" --shard "$shard" \
        >"$run/$part.shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    test "$status" -eq 0
  done
  "$python" "$root/tools/verify_multiview_event_replay.py" \
    "${verify[@]}" --shards "$shards" \
    >"$run/verification.log" 2>&1
fi
test -s "$run/verification.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
