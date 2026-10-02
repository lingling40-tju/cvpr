#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference/seed33_novel"
out="$run/swaps"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock -n 9 || { echo 'seed33 novel swap cache already running' >&2; exit 2; }
if test -f "$out/suite.completed"; then exit 0; fi
rm -f "$out/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$out/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
for _ in $(seq 1 240); do
  if test -f "$run/suite.completed"; then break; fi
  if test -f "$run/suite.failed"; then echo 'novel frame collection failed' >&2; exit 1; fi
  sleep 30
done
test -f "$run/suite.completed"
test "$(sha256sum "$run/instruction_swaps.json" | awk '{print $1}')" = \
  0a0cc6c8f332e953c5b57c452c23fa6d7b68d1cec5390c66267847396ad2219e
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/cache_temporal_contrastive_swaps.py \
  --manifest "$run/manifest.json" --v2-manifest "$run/instruction_swaps.json" \
  --records-root "$run/frames" \
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --output-root "$out" --split audit >"$out/cache.log" 2>&1
"$base/activevln_train_env/bin/python" - "$out/summary_audit.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['records']==38 and data['split']=='audit'
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/suite.completed"
