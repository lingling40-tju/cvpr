#!/usr/bin/env bash
set -euo pipefail
arm=${1:?pass control or candidate}
case "$arm" in control|candidate) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
label="positive_trajectory_${arm}_64step_seed11"
checkpoint="$root/verl_checkpoints/$label/global_step_64/actor/huggingface"
result="$root/runlogs/positive_development256"
manifest="$root/prepared_data/development256.json"
port=8135
mkdir -p "$result" "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu0.lock"
flock 8
exec 9>"$result/$label.lock"
flock 9
test -f "$root/runlogs/$label/completed"
test -f "$checkpoint/config.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
if test -f "$result/$label.completed"; then exit 0; fi
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "port $port is occupied" >&2
  exit 1
fi
mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$mem0" -lt 10000 && test "$mem2" -lt 10000
rm -f "$result/$label.failed"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$base/activevln_train_env/bin/vllm" serve "$checkpoint" --port "$port" \
  --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  --disable-log-requests --seed 11 >"$result/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  status=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$result/$label.failed"; fi
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
test "$ready" -eq 1
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 "$base/activevln_server_env/bin/python" \
    tools/eval_train_scene_subset.py --model-label "$label" \
    --manifest "$manifest" --result-root "$result" --role development \
    --count 256 --shard-count 4 --shard-index "$shard" --max-turns 12 \
    --base-url "http://127.0.0.1:$port/v1" \
    >"$result/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
"$base/activevln_server_env/bin/python" tools/validate_train_label.py \
  --root "$result" --manifest "$manifest" --role development \
  --label "$label" --output "$result/$label.validated.json" \
  >"$result/$label.validation.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/$label.completed"
