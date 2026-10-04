#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode" in smoke|full) ;; *) echo 'usage: run_anchor_potential_lora.sh smoke|full' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
manifest="$root/runlogs/future_advantage_sparse_manifest/replay_manifest.json"
labels="$root/runlogs/future_advantage_sparse_manifest/pair_labels.json"
report="$root/runlogs/future_advantage_pooled_preflight/report.json"
preflight="$root/runlogs/dense_potential_preflight/report.json"
replay="$root/runlogs/future_advantage_sparse_replay/full"
run="$root/runlogs/anchor_distance_potential_lora/$mode"
python="$base/activevln_train_env/bin/python"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "anchor potential $mode already active" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$replay/completed" && test -s "$replay/verification.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  4a0a2403fc4345308545d36f17102664856e24129eab189a32c0478a5bcf7967
test "$(sha256sum "$preflight" | awk '{print $1}')" = \
  96211bd5efca303526c7181d4b12fe6e11132965793a6d3ae370d8112439c22e
if test "$mode" = full; then
  test -f "$root/runlogs/anchor_distance_potential_lora/smoke/completed"
fi
# The evaluator and other reward fits acquire this lock before GPU-1 work.
eval_lock="$base/ActiveVLN_three_directions_20261002/runlogs/gpu_eval_locks/gpu1.lock"
mkdir -p "$(dirname "$eval_lock")"
exec 8>"$eval_lock"
flock 8
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 occupied ($used MiB)" >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$root"
sha256sum tools/train_anchor_potential_lora.py \
  tools/anchor_potential_visual_input.py \
  tools/train_future_advantage_sparse_lora.py \
  tools/future_advantage_visual_input.py \
  tools/verify_future_advantage_sparse_replay.py \
  "$model/config.json" "$manifest" "$labels" "$report" \
  "$preflight" "$replay/verification.json" >"$run/source.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
if test "$mode" = smoke; then
  "$python" tools/train_anchor_potential_lora.py --model "$model" \
    --synthetic-smoke >"$run/train.log" 2>&1
  grep -F '"schema": "anchor_potential_synthetic_smoke_v1"' \
    "$run/train.log" >/dev/null
else
  "$python" tools/train_anchor_potential_lora.py --model "$model" \
    --manifest "$manifest" --labels "$labels" --report "$report" \
    --preflight "$preflight" --replay-root "$replay/rgb" \
    --output "$run" >"$run/train.log" 2>&1
  test -s "$run/source_verification.json" && test -s "$run/adapter_head.pt"
  test -s "$run/development.json"
  "$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
report=json.load(open(sys.argv[1]))
run=pathlib.Path(sys.argv[2])
assert report['schema']=='anchor_distance_potential_development_v1'
assert report['fit_microsteps']==1024
gate=report['development']['development_gate']
assert set(gate)=={'3','6'}
passed=bool(report['development']['passed'] and all(gate.values()))
(run/('passed_development_gate' if passed else 'failed_development_gate')).write_text(
    'exploratory train-scene development; no navigation claim\n')
print(json.dumps({'passed':passed,'anchors':gate}),flush=True)
PY
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
