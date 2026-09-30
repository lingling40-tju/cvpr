#!/usr/bin/env bash
set -euo pipefail

# Usage: bash tools/run_full_val_unseen.sh LABEL MODEL_PATH
label=${1:?label required}
model=${2:?model checkpoint path required}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
python_env="$base/activevln_train_env"
sim_env="$base/activevln_server_env"
result_root="$root/runlogs/eventtrace_full_val_unseen"
manifest="$result_root/manifest.json"
mkdir -p "$result_root"
cd "$root"
export PYTHONPATH="$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$manifest" ]; then
  "$sim_env/bin/python" tools/prepare_full_val_manifest.py "$manifest"
fi
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$python_env/bin/vllm" serve "$model" --port 8004 \
  --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  >"$result_root/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "vLLM port 8004 remained open after $label" >&2
  return 1
}
trap cleanup EXIT

ready=0
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then echo "vLLM failed for $label" >&2; exit 1; fi

for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 "$sim_env/bin/python" tools/eval_val_unseen_subset.py \
    --model-label "$label" --manifest "$manifest" --result-root "$result_root" \
    --all-episodes --shard-count 4 --shard-index "$shard" --max-turns 12 \
    >"$result_root/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then echo "evaluation failed for $label" >&2; exit 1; fi
echo "full val_unseen evaluation finished for $label"
