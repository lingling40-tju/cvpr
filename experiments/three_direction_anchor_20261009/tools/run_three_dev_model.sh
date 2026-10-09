#!/usr/bin/env bash
set -Eeuo pipefail
method=${1:?method}
gpu=${2:?GPU0 or GPU1}
freeze_sha=${3:?evaluation identity SHA}
case "$method" in grpo_anchor|turn_rloo|srgpo) ;; *) exit 2 ;; esac
case "$gpu" in 0) port=8151 ;; 1) port=8152 ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_direction_20261009"
label="td_${method}_n4_seed11"
checkpoint="$root/verl_checkpoints/$label/global_step_64/actor/huggingface"
result="$root/runlogs/development256"
manifest="$root/prepared_data/development256.json"
mkdir -p "$result" "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu${gpu}.lock"
flock 8
exec 9>"$result/$label.lock"
flock 9
cd "$root"
test -f "$root/runlogs/sequential_training/training.completed"
test -f "$root/runlogs/td_${method}_n4_seed11/completed"
test ! -f "$root/runlogs/td_${method}_n4_seed11/failed"
test -f "$root/runlogs/development_suite/${method}_train_audit.json"
test -f "$checkpoint/model.safetensors.index.json"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_three_eval_freeze.py --root "$root" --identity-sha "$freeze_sha"
if test -f "$result/$label.completed"; then
  test ! -f "$result/$label.failed"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest "$manifest" --role development --label "$label" --output "$result/$label.validated.json"
  exit 0
fi
test ! -f "$result/$label.failed"
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then echo "port occupied: $port" >&2; exit 1; fi
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
test "$memory" -lt 10000
if curl -fsS --max-time 1 http://127.0.0.1:5086/health >/dev/null 2>&1; then echo 'training service still running' >&2; exit 1; fi
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$base/activevln_train_env/bin/vllm" serve "$checkpoint" --port "$port" \
  --dtype half --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  --disable-log-requests --seed 11 >"$result/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  rc=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if test -n "$server_pid"; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$result/$label.failed"; fi
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
test "$ready" -eq 1
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py --log "$result/vllm_${label}.log" --checkpoint "$checkpoint" --expected-dtype float16 --output "$result/$label.ready_dtype.json"
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 "$base/activevln_server_env/bin/python" tools/eval_train_scene_subset.py \
    --model-label "$label" --manifest "$manifest" --result-root "$result" --role development \
    --count 256 --shard-count 4 --shard-index "$shard" --max-turns 12 --base-url "http://127.0.0.1:$port/v1" \
    >"$result/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
rc=0
for pid in "${worker_pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest "$manifest" --role development --label "$label" --output "$result/$label.validated.json" >"$result/$label.validation.log"
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py --log "$result/vllm_${label}.log" --checkpoint "$checkpoint" --expected-dtype float16 --output "$result/$label.dtype.json"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_three_eval_freeze.py --root "$root" --identity-sha "$freeze_sha"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/$label.completed"
