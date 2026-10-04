#!/usr/bin/env bash
set -euo pipefail

# The fixed final checkpoint is the only development-scored checkpoint.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
source_run="$root/runlogs/future_advantage_sparse_manifest"
replay="$root/runlogs/future_advantage_sparse_replay/full"
run="$root/runlogs/future_advantage_sparse_lora"
manifest="$source_run/replay_manifest.json"
labels="$source_run/pair_labels.json"
report="$root/runlogs/future_advantage_pooled_preflight/report.json"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
python="$base/activevln_train_env/bin/python"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'sparse future-advantage fit already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed" "$run/passed_development_gate" "$run/failed_development_gate"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$source_run/completed" && test -f "$replay/completed"
test -s "$manifest" && test -s "$labels" && test -s "$report"
test -s "$replay/verification.json"
test -f "$root/runlogs/oracle_candidate_eval_overlap/completed"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  sed -n 2p | tr -d ' ')
test -n "$used" && test "$used" -lt 5000 || {
  echo "GPU 1 occupied ($used MiB); defer LoRA fit" >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$root/tools:$root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$root"
sha256sum tools/train_future_advantage_sparse_lora.py \
  tools/future_advantage_visual_input.py \
  tools/future_advantage_live_prefix.py \
  tools/verify_future_advantage_sparse_replay.py \
  "$model/config.json" "$manifest" "$labels" "$report" \
  "$replay/verification.json" >"$run/source.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"
"$python" tools/train_future_advantage_sparse_lora.py \
  --model "$model" --manifest "$manifest" --labels "$labels" \
  --report "$report" --replay-root "$replay/rgb" --output "$run" \
  >"$run/train.log" 2>&1
test -s "$run/source_verification.json"
test -s "$run/adapter_head.pt" && test -s "$run/development.json"
"$python" - "$run/development.json" "$run" <<'PY'
import json,pathlib,sys
metrics=json.load(open(sys.argv[1]))
run=pathlib.Path(sys.argv[2])
assert metrics['schema']=='future_advantage_sparse_development_v1'
assert metrics['fit_microsteps']==1024
decision=metrics['development']['development_gate']
assert set(decision)=={'3','6'}
name='passed_development_gate' if metrics['development']['passed'] and all(
    decision.values()) else 'failed_development_gate'
(run/name).write_text('train-scene development only; no navigation claim\n')
print(json.dumps({'gate':name,'anchors':decision}),flush=True)
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
