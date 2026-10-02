#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/failure_rank"
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9 || { echo 'failure-rank encoder training already active' >&2; exit 2; }
if test -f "$run/training.completed"; then exit 0; fi
rm -f "$run/training.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/training.failed"; fi
}
trap on_exit EXIT
until test -f "$run/cache.completed"; do
  if test -f "$run/cache.failed"; then echo 'failure-rank feature cache failed' >&2; exit 1; fi
  pid=$(cat "$run/cache_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "cache PID $pid stopped" >&2; exit 1; }
  sleep 30
done
cd "$source_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$root/tools:$source_root/tools:$source_root/vlnce_server:$source_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
nice -n 10 "$base/activevln_train_env/bin/python" \
  "$root/tools/train_failure_rank_encoder.py" \
  --manifest "$run/manifest.json" \
  --features "$run/features_full/features.pt" \
  --old-encoder "$source_root/runlogs/ordinal_progress/policy_preference/temporal_contrastive_v2/encoder.pt" \
  --output "$run/development.json" --checkpoint "$run/encoder.pt" \
  --seed 11 --epochs 30 >"$run/training.log" 2>&1
"$base/activevln_train_env/bin/python" - "$run/development.json" "$run/encoder.pt" <<'PY'
import json,sys,torch
report=json.load(open(sys.argv[1]))
checkpoint=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert report['schema']=='failure_rank_encoder_development_v1'
assert report['manifest_sha256']==checkpoint['manifest_sha256']=='139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0'
assert report['baseline_development']['pairs']==report['development']['pairs']==125
assert report['fit']['pairs']==697
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"
