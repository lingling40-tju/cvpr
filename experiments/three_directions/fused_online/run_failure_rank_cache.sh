#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/failure_rank"
mkdir -p "$run"
exec 9>"$run/cache.lock"
flock -n 9 || { echo 'failure-rank feature cache already running' >&2; exit 2; }
if test -f "$run/cache.completed"; then exit 0; fi
rm -f "$run/cache.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/cache.failed"; fi
}
trap on_exit EXIT
until test -f "$run/replay.completed"; do
  if test -f "$run/replay.failed"; then echo 'failure-rank replay failed' >&2; exit 1; fi
  pid=$(cat "$run/replay_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "replay PID $pid stopped" >&2; exit 1; }
  sleep 30
done
test "$(sha256sum "$run/manifest.json" | awk '{print $1}')" = \
  139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0
cd "$source_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$root/tools:$source_root/vlnce_server:$source_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nice -n 10 "$base/activevln_train_env/bin/python" \
  "$root/tools/cache_failure_rank_sft.py" \
  --manifest "$run/manifest.json" \
  --records-root "$run/replay_full" \
  --model "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --output-root "$run/features_full" \
  >"$run/cache.log" 2>&1
"$base/activevln_train_env/bin/python" - "$run/features_full/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['records'] == 1856 and summary['frames'] == 7424
assert summary['manifest_sha256'] == '139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0'
assert summary['model_config_sha256'] == '9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9'
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/cache.completed"
