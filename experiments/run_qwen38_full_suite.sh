#!/usr/bin/env bash
set -euo pipefail

# Exact 1,839-episode val-unseen evaluation for the six 256-step checkpoints.
# Reuse the original SFT evaluation; its manifest and decoding are unchanged.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
train="$root/runlogs/eventtrace_qwen38_r2r256"
old="$root/runlogs/eventtrace_full_val_unseen"
out="$root/runlogs/eventtrace_qwen38_full_val_unseen"
mkdir -p "$out"
exec 9>"$out/suite.lock"
flock 9

for seed in 11 22 33; do
  for arm in control event; do
    label="seed${seed}_${arm}"
    test -f "$train/$label.completed" || {
      echo "training not complete: $label" >&2
      exit 1
    }
  done
done

if [ ! -f "$out/manifest.json" ]; then
  cp "$old/manifest.json" "$out/manifest.json"
fi
cmp "$old/manifest.json" "$out/manifest.json"
if [ ! -e "$out/sft" ]; then
  ln -s ../eventtrace_full_val_unseen/sft "$out/sft"
fi
cp "$old/sft.completed" "$old/sft.validated.json" "$out/"

for seed in 11 22 33; do
  for arm in control event; do
    label="seed${seed}_${arm}"
    if [ -f "$out/$label.completed" ]; then continue; fi
    model="$root/verl_checkpoints/eventtrace_qwen38_r2r256_${label}/global_step_256/actor/huggingface"
    test -f "$model/config.json"
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/$label.started"
    rm -f "$out/$label.failed"
    if EVENTTRACE_FULL_RESULT_ROOT="$out" EVENTTRACE_INFERENCE_GPU=1 \
      EVENTTRACE_SIM_GPU=2 EVENTTRACE_INFERENCE_PORT=8004 EVENTTRACE_SHARDS=4 \
      bash "$root/tools/run_full_val_unseen.sh" "$label" "$model" \
      >"$out/$label.run.log" 2>&1; then
      test -f "$out/$label.validated.json"
      test -f "$out/$label.completed"
    else
      status=$?
      printf '%s\n' "$status" >"$out/$label.failed"
      exit "$status"
    fi
  done
done

"$base/activevln_server_env/bin/python" "$root/tools/analyze_full_val.py" \
  --root "$out" >"$out/analysis.log" 2>&1
python3 "$root/tools/export_full_val_compact.py" "$out" "$out/episodes.csv" \
  >"$out/export.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$out/suite.completed"
