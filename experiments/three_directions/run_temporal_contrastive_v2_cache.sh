#!/usr/bin/env bash
set -euo pipefail

# Re-encode existing train RGB under fixed same-scene wrong instructions.
# The new audit partition is deliberately untouched at this stage.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
v2="$run/temporal_contrastive_v2"
out="$v2/swaps"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock -n 9 || { echo 'v2 swapped-feature cache already running' >&2; exit 2; }
if test -f "$out/fit_development.completed"; then exit 0; fi
rm -f "$out/fit_development.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$out/fit_development.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$run/collection.completed"
test "$(sha256sum "$v2/manifest.json" | awk '{print $1}')" = \
  07545fc6d55510e9d89d7e72372e11104425d34ce28518d5ef848971be1a4ac5
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
test "$(sha256sum "$model/config.json" | awk '{print $1}')" = \
  9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9
export CUDA_VISIBLE_DEVICES=0
for split in fit development; do
  "$base/activevln_train_env/bin/python" tools/cache_temporal_contrastive_swaps.py \
    --manifest "$run/manifest.json" --v2-manifest "$v2/manifest.json" \
    --records-root "$run/frames_full" --model "$model" \
    --output-root "$out" --split "$split" \
    >"$out/cache_${split}.log" 2>&1
done
"$base/activevln_train_env/bin/python" - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for split,count in [('fit',300),('development',52)]:
 data=json.load(open(root/f'summary_{split}.json'))
 assert data['split']==split and data['records']==count
assert len(list((root/'records').glob('*.pt')))==352
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/fit_development.completed"
