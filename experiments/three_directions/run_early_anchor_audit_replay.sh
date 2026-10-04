#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_early_anchor_audit_replay.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
scratch="$root/runlogs/early_anchor_audit"
manifest="$scratch/source_manifest.json"
run="$scratch/$mode"
python="$base/activevln_server_env/bin/python"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "early anchor audit $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1
if test "$mode" = full; then test -f "$scratch/smoke/completed"; fi
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu1.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 occupied ($used MiB)" >&2; exit 1;
}
export PYTHONPATH="$root/tools:$root:$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
if test "$mode" = smoke; then
  "$python" tools/collect_early_anchor_audit_frames.py \
    --manifest "$manifest" --root "$root" \
    --output-root "$scratch/rgb" --gpu 1 --shards 1 --shard 0 \
    --limit 1 >"$run/collect.log" 2>&1
  test -s "$scratch/rgb/audit/summary.shard0.json"
else
  pids=()
  for shard in 0 1 2 3; do
    "$python" tools/collect_early_anchor_audit_frames.py \
      --manifest "$manifest" --root "$root" \
      --output-root "$scratch/rgb" --gpu 1 --shards 4 --shard "$shard" \
      >"$run/shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  test "$status" -eq 0
  "$python" tools/verify_early_anchor_audit_replay.py \
    --manifest "$manifest" --rgb-root "$scratch/rgb" \
    --output "$run/verification.json" >"$run/verification.log" 2>&1
  test -s "$run/verification.json"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
