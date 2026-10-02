#!/usr/bin/env bash
set -euo pipefail

# Recompute the frozen state with VLNCE's exact single-observation templates.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference"
out="$run/navigation_sft_server_prompt"
mkdir -p "$out"
exec 9>"$run/navigation_sft_server_prompt.lock"
flock -n 9 || { echo 'server-prompt SFT probe already running' >&2; exit 2; }
if test -f "$run/navigation_sft_server_prompt.completed"; then exit 0; fi
rm -f "$run/navigation_sft_server_prompt.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/navigation_sft_server_prompt.failed"; fi
}
trap on_exit EXIT
cd "$root"
test -f "$run/collection.completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  cc3cb63c0ae1ae3c9b47a9d191dbf548d9dce12e08feed0ec8b314a1b7398c47
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
test "$(sha256sum "$model/config.json" | awk '{print $1}')" = \
  9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9
export CUDA_VISIBLE_DEVICES=0
"$base/activevln_train_env/bin/python" tools/cache_navigation_sft_state.py \
  --manifest "$run/manifest.json" --records-root "$run/frames_full" \
  --model "$model" --output-root "$out" >"$run/navigation_sft_server_prompt.log" 2>&1
"$base/activevln_train_env/bin/python" - "$out/summary.json" "$out/features.pt" <<'PY'
import json,sys,torch
summary=json.load(open(sys.argv[1]))
features=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert summary['records']==800 and summary['frames']==3200
assert summary['prompt_version']=='vlnce_server_single_observation_v1'
assert features['hidden'].shape==(800,4,2048)
assert features['stop_margin'].shape==(800,4)
assert torch.isfinite(features['hidden']).all()
assert torch.isfinite(features['stop_margin']).all()
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/navigation_sft_server_prompt.completed"
