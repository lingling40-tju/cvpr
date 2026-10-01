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
inference_gpu=${EVENTTRACE_INFERENCE_GPU:-1}
sim_gpu=${EVENTTRACE_SIM_GPU:-2}
port=${EVENTTRACE_INFERENCE_PORT:-8004}
shards=${EVENTTRACE_SHARDS:-4}
mkdir -p "$result_root"
cd "$root"
export PYTHONPATH="$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$result_root/$label.lock"
flock 9
if [ -f "$result_root/$label.completed" ]; then
  echo "$label already completed"
  exit 0
fi

if [ ! -f "$manifest" ]; then
  "$sim_env/bin/python" tools/prepare_full_val_manifest.py "$manifest"
fi
if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
  echo "port $port is already serving a model; refusing ambiguous evaluation" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$inference_gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$python_env/bin/vllm" serve "$model" --port "$port" \
  --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  >"$result_root/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "vLLM port $port remained open after $label" >&2
  return 1
}
trap cleanup EXIT

ready=0
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then echo "vLLM failed for $label" >&2; exit 1; fi

for ((shard=0; shard<shards; shard++)); do
  CUDA_VISIBLE_DEVICES="$sim_gpu" "$sim_env/bin/python" tools/eval_val_unseen_subset.py \
    --model-label "$label" --manifest "$manifest" --result-root "$result_root" \
    --all-episodes --shard-count "$shards" --shard-index "$shard" --max-turns 12 \
    --base-url "http://127.0.0.1:$port/v1" \
    >"$result_root/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then echo "evaluation failed for $label" >&2; exit 1; fi
"$sim_env/bin/python" tools/validate_full_label.py "$label" "$result_root" "$manifest" "$shards" \
  >"$result_root/$label.validated.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result_root/$label.completed"
echo "full val_unseen evaluation finished for $label"
