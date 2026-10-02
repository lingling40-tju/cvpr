#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
s="$base/ActiveVLN_three_directions_20261002"
r="$base/ActiveVLN_fused_reward_20261003"
p="$s/runlogs/ordinal_progress/policy_preference"
run="$r/runlogs/failure_rank/multitask"
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9 || { echo 'multitask rank training already active' >&2; exit 2; }
if test -f "$run/training.completed"; then exit 0; fi
rm -f "$run/training.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/training.failed"; fi
}
trap on_exit EXIT
test -f "$r/runlogs/fused_reward_group4_64step_seed11/completed"
cd "$s"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$r/tools:$s/tools:$s${PYTHONPATH:+:$PYTHONPATH}"
nice -n 10 "$base/activevln_train_env/bin/python" \
  "$r/tools/train_multitask_rank_encoder.py" \
  --failure-manifest "$r/runlogs/failure_rank/manifest.json" \
  --failure-features "$r/runlogs/failure_rank/features_full/features.pt" \
  --old-pair-manifest "$p/manifest.json" \
  --old-v2-manifest "$p/temporal_contrastive_v2/manifest.json" \
  --old-features "$p/navigation_sft_server_prompt/features.pt" \
  --old-swaps "$p/temporal_contrastive_v2/swaps" \
  --old-encoder "$p/temporal_contrastive_v2/encoder.pt" \
  --epochs 20 --seed 11 \
  --output "$run/development.json" \
  --checkpoint "$run/encoder.pt" >"$run/train.log" 2>&1
test -s "$run/development.json"
"$base/activevln_train_env/bin/python" - "$run/development.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['schema']=='multitask_failure_success_grounding_development_v1'
assert len(report['history'])==20
print('selected_epoch',report['selected_epoch'])
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"
