#!/usr/bin/env bash
set -euo pipefail

label=${1:?model label required}
model=${2:?absolute checkpoint path required}
inference_gpu=${3:-1}
sim_gpu=${4:-2}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
result_root=${VLN_EVAL_RESULT_ROOT:-$root/runlogs/three_direction_val256}
manifest=${VLN_EVAL_MANIFEST:-$result_root/manifest.json}
episode_count=${VLN_EVAL_COUNT:-256}
shard_count=${VLN_EVAL_SHARDS:-4}
port=${VLN_EVAL_PORT:-8010}
[[ "$episode_count" =~ ^[1-9][0-9]*$ ]]
[[ "$shard_count" =~ ^[1-9][0-9]*$ ]]
[[ "$port" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$result_root"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
# The overlap watcher and the final paired suite may evaluate different
# labels on the same inference GPU. Serialize them before taking the
# per-label lock so they cannot start two vLLM servers on one GPU.
mkdir -p "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu${inference_gpu}.lock"
flock 8
exec 9>"$result_root/$label.lock"
flock 9
test -f "$manifest"
test -f "$model/config.json"
if [ -f "$result_root/$label.completed" ]; then exit 0; fi
if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
  echo "port $port already in use" >&2
  exit 1
fi
rm -f "$result_root/$label.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result_root/$label.started"
if [ -n "${VLN_VLLM_SEED:-}" ]; then
  [[ "$VLN_VLLM_SEED" =~ ^[0-9]+$ ]]
  CUDA_VISIBLE_DEVICES="$inference_gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
    "$base/activevln_train_env/bin/vllm" serve "$model" --port "$port" \
    --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
    --disable-log-requests \
    --seed "$VLN_VLLM_SEED" >"$result_root/vllm_${label}.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES="$inference_gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
    "$base/activevln_train_env/bin/vllm" serve "$model" --port "$port" \
    --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
    --disable-log-requests \
    >"$result_root/vllm_${label}.log" 2>&1 &
fi
server_pid=$!
worker_pids=()
cleanup() {
  status=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$result_root/$label.failed"; fi
}
trap cleanup EXIT
ready=0
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
test "$ready" -eq 1
for ((shard=0; shard<shard_count; shard++)); do
  CUDA_VISIBLE_DEVICES="$sim_gpu" "$base/activevln_server_env/bin/python" \
    tools/eval_val_unseen_subset.py --model-label "$label" \
    --manifest "$manifest" --result-root "$result_root" --count "$episode_count" \
    --shard-count "$shard_count" --shard-index "$shard" --max-turns 12 \
    --base-url "http://127.0.0.1:$port/v1" \
    >"$result_root/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
"$base/activevln_server_env/bin/python" tools/validate_full_label.py \
  "$label" "$result_root" "$manifest" "$shard_count" >"$result_root/$label.validated.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result_root/$label.completed"
