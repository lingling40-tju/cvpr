#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
python_env="$base/activevln_train_env"
sim_env="$base/activevln_server_env"
result_root="$root/runlogs/cvpr_val_unseen_subset"
manifest="$result_root/manifest.json"
mkdir -p "$result_root"
cd "$root"
export PYTHONPATH="$root/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"

for label in sft control event; do
  case "$label" in
    sft) model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" ;;
    control) model="$root/verl_checkpoints/semantic_pilot8_control_20260930/global_step_2/actor/huggingface" ;;
    event) model="$root/verl_checkpoints/semantic_pilot8_event_20260930/global_step_2/actor/huggingface" ;;
  esac
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
    "$python_env/bin/vllm" serve "$model" \
    --port 8004 --max-model-len 16384 \
    --gpu-memory-utilization 0.72 --trust-remote-code \
    >"$result_root/vllm_${label}.log" 2>&1 &
  server_pid=$!
  ready=0
  for attempt in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 3
  done
  if [ "$ready" -ne 1 ]; then
    kill "$server_pid" 2>/dev/null || true
    echo "vLLM failed to become ready for $label" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=2 "$sim_env/bin/python" tools/eval_val_unseen_subset.py \
    --model-label "$label" --manifest "$manifest" \
    --result-root "$result_root" --count 16 --max-turns 12 \
    >"$result_root/eval_${label}.log" 2>&1
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
done
