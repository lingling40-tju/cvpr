#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
temporal="$run/temporal_progress"
out="$temporal/swaps"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock -n 9 || { echo 'temporal swap audit already running' >&2; exit 2; }
if test -f "$out/suite.completed"; then exit 0; fi
rm -f "$out/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$out/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$temporal/suite.completed"
test "$(sha256sum "$temporal/instruction_swaps.json" | awk '{print $1}')" = \
  823f5754e1e9503bf05fed857f1eae0d6daa7f821b405487e2032e0b6211918c
"$base/activevln_train_env/bin/python" - "$temporal/screen.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert all(data['predeclared_gate'].values())
assert data['audit']['pairs']==54
PY
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/audit_temporal_instruction_swaps.py \
  --manifest "$run/manifest.json" --swaps "$temporal/instruction_swaps.json" \
  --records-root "$run/frames_full" \
  --features "$run/navigation_sft_server_prompt/features.pt" \
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --encoder "$temporal/encoder.pt" --output-root "$out" \
  >"$out/audit.log" 2>&1
"$base/activevln_train_env/bin/python" - "$out/summary.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['pairs']==54 and data['scenes']==8
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/suite.completed"
