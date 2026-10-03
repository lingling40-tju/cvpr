#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/stop_readiness_probe"
mkdir -p "$run"
exec 9>"$run/visual.lock"
flock -n 9 || { echo 'STOP visual probe already running' >&2; exit 2; }
if test -f "$run/visual.completed"; then exit 0; fi
rm -f "$run/visual.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/visual.failed"; fi
}
trap on_exit EXIT
test -f "$run/collection.completed"
cd "$root"
CUDA_VISIBLE_DEVICES=2 "$base/activevln_train_env/bin/python" \
  tools/score_stop_readiness_visual.py \
  --manifest "$run/manifest.json" --records-root "$run/frames" \
  --model "$base/models/siglip-base-patch16-224" \
  --adapter "$source_root/runlogs/ordinal_progress/siglip_lora_goal_policy_fixed64_768_seed11/adapter.pt" \
  --output "$run/visual_scores.json" >"$run/visual.log" 2>&1
python - "$run/visual_scores.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert len(x['records'])==531
assert len({row['record_id'] for row in x['records']})==531
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/visual.completed"
