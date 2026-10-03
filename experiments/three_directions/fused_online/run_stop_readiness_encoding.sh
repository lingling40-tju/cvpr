#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_stopaware_20261003"
run="$root/runlogs/stop_readiness_probe"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/encoding.lock"
flock -n 9 || { echo 'STOP probe encoding already running' >&2; exit 2; }
if test -f "$run/encoding.completed"; then exit 0; fi
rm -f "$run/encoding.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/encoding.failed"; fi
}
trap on_exit EXIT
test -f "$run/collection.completed"
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  ec53e58c04e8b1e60a5c19c052ef5d726a44c75ea8ec82d94a70c59c9fcd02ad
test "$(sha256sum "$model/config.json" | awk '{print $1}')" = \
  9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9
cd "$root"
CUDA_VISIBLE_DEVICES=1 "$base/activevln_train_env/bin/python" \
  tools/cache_stop_readiness_states.py --manifest "$run/manifest.json" \
  --records-root "$run/frames" --model "$model" \
  --output-root "$run/features_full" >"$run/encoding.log" 2>&1
"$base/activevln_train_env/bin/python" - "$run/features_full/summary.json" \
  "$run/features_full/features.pt" <<'PY'
import json,sys,torch
summary=json.load(open(sys.argv[1]))
cache=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
assert summary['records']==531 and summary['frames']==1062
assert cache['hidden'].shape==(531,2,2048)
assert cache['stop_margin'].shape==(531,2)
assert torch.isfinite(cache['hidden']).all()
assert torch.isfinite(cache['stop_margin']).all()
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/encoding.completed"
