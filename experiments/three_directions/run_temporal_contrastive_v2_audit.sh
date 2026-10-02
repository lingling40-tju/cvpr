#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
v2="$run/temporal_contrastive_v2"
mkdir -p "$v2"
exec 9>"$v2/audit.lock"
flock -n 9 || { echo 'temporal v2 audit already running' >&2; exit 2; }
if test -f "$v2/audit.completed"; then exit 0; fi
rm -f "$v2/audit.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$v2/audit.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$v2/training.completed"
"$base/activevln_train_env/bin/python" - "$v2/development.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert all(data['predeclared_gate'].values())
PY
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/cache_temporal_contrastive_swaps.py \
  --manifest "$run/manifest.json" --v2-manifest "$v2/manifest.json" \
  --records-root "$run/frames_full" \
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --output-root "$v2/swaps" --split audit \
  >"$v2/cache_audit.log" 2>&1
"$base/activevln_train_env/bin/python" tools/audit_temporal_contrastive_v2.py \
  --manifest "$run/manifest.json" --v2-manifest "$v2/manifest.json" \
  --features "$run/navigation_sft_server_prompt/features.pt" \
  --labels-root "$run/geodesic_labels" --swaps-root "$v2/swaps" \
  --development "$v2/development.json" --checkpoint "$v2/encoder.pt" \
  --output "$v2/audit.json" >"$v2/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$v2/audit.completed"
