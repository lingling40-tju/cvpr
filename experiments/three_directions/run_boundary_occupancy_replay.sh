#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_boundary_occupancy_replay.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_dir="$root/runlogs/boundary_occupancy_source"
manifest="$source_dir/replay_manifest.json"
labels="$source_dir/privileged_labels.json"
preflight="$source_dir/preflight.json"
scratch="$root/runlogs/boundary_occupancy_rgb"
run="$scratch/$mode"
python="$base/activevln_server_env/bin/python"
gpu=${VLN_BOUNDARY_GPU:-1}
shards=${VLN_BOUNDARY_SHARDS:-4}
[[ "$gpu" =~ ^[0-3]$ && "$shards" =~ ^[1-4]$ ]]
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "boundary $mode replay already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -s "$manifest" && test -s "$labels" && test -s "$preflight"
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
    echo "GPU $gpu occupied ($used MiB); defer RGB replay" >&2
    return 1
  }
}

if test "$mode" = smoke; then
  check_gpu
  rid=s11_e5986_v2
  "$python" "$root/tools/collect_boundary_occupancy_frames.py" \
    --manifest "$manifest" --preflight "$preflight" --root "$root" \
    --part fit --output-root "$run/rgb" --gpu "$gpu" \
    --record-id "$rid" >"$run/collect.log" 2>&1
  "$python" "$root/tools/verify_boundary_occupancy_replay.py" \
    --manifest "$manifest" --labels "$labels" --preflight "$preflight" \
    --replay-root "$run/rgb" --smoke-record-id "$rid" \
    --output "$run/verification.json" >"$run/verification.log" 2>&1
else
  test -f "$scratch/smoke/completed" && test -s "$scratch/smoke/verification.json"
  "$python" - "$scratch/smoke/verification.json" "$manifest" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1]))
assert x['manifest_sha256']==hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
assert x['parts'][0]['records']==1 and x['parts'][0]['rgb_images']==2
PY
  for part in fit development; do
    check_gpu
    pids=()
    for ((shard=0; shard<shards; shard++)); do
      "$python" "$root/tools/collect_boundary_occupancy_frames.py" \
        --manifest "$manifest" --preflight "$preflight" --root "$root" \
        --part "$part" --output-root "$run/rgb" --gpu "$gpu" \
        --shards "$shards" --shard "$shard" \
        >"$run/$part.shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    test "$status" -eq 0
  done
  "$python" "$root/tools/verify_boundary_occupancy_replay.py" \
    --manifest "$manifest" --labels "$labels" --preflight "$preflight" \
    --replay-root "$run/rgb" --shards "$shards" \
    --output "$run/verification.json" >"$run/verification.log" 2>&1
fi
test -s "$run/verification.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
