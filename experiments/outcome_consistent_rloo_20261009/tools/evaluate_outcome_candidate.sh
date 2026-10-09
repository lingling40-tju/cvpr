#!/usr/bin/env bash
set -Eeuo pipefail
identity_sha=${1:?frozen identity sha}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_outcome_consistent_rloo_20261009"
label=td_outcome_consistent_rloo_n4_seed11
checkpoint="$root/verl_checkpoints/$label/global_step_64/actor/huggingface"
result="$root/runlogs/development256"
state="$root/runlogs/development_suite"
manifest="$root/prepared_data/development256.json"
port=8153
mkdir -p "$result" "$state" "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu0.lock"
flock 8
exec 9>"$state/evaluation.lock"
flock -n 9 || { echo 'evaluation already active' >&2; exit 2; }
cd "$root"
test -f "$root/runlogs/sequential_training/training.completed"
test -f "$root/runlogs/$label/completed"
test ! -f "$root/runlogs/$label/failed"
test -f "$state/outcome_consistent_rloo_train_audit.json"
test -f "$checkpoint/model.safetensors.index.json"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_outcome_freeze.py --root "$root" --identity-sha "$identity_sha" >"$state/freeze_before_inference.json"
test ! -f "$state/development.opened"
test ! -f "$state/suite.completed"
test ! -f "$state/suite.failed"
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then echo "port occupied: $port" >&2; exit 1; fi
memory=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$memory" -lt 10000
if curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then echo 'owned Habitat service still running' >&2; exit 1; fi
if test ! -L "$result/positive_initial_sft" || ! test -L "$result/td_turn_rloo_n4_seed11"; then
  bash tools/prepare_outcome_refs.sh >"$state/prepare_refs.log" 2>&1
fi
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest "$manifest" --role development --label positive_initial_sft --output "$result/positive_initial_sft.validated.json" >"$state/sft_reuse_validation.log"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest "$manifest" --role development --label td_turn_rloo_n4_seed11 --output "$result/td_turn_rloo_n4_seed11.validated.json" >"$state/previous_rloo_reuse_validation.log"
 CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py --log "$base/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_development/vllm_positive_initial_sft.log" --checkpoint "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" --expected-dtype float16 --output "$result/positive_initial_sft.dtype.json" >"$state/sft_reuse_dtype.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/development.opened"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$base/activevln_train_env/bin/vllm" serve "$checkpoint" --port "$port" \
  --dtype half --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  --disable-log-requests --seed 11 >"$result/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  rc=$?
  for pid in "${worker_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  if test -n "${server_pid:-}"; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$state/suite.failed"; fi
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
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_outcome_freeze.py --root "$root" --identity-sha "$identity_sha" >"$state/freeze_after_inference.json"
for spec in 'outcome_vs_sft:positive_initial_sft' 'outcome_vs_previous_rloo:td_turn_rloo_n4_seed11'; do
  name=${spec%%:*}; control=${spec#*:}
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py --root "$result" --manifest "$manifest" --role development --control "$control" --candidate "$label" --compact "$state/$name.jsonl" --output "$state/$name.json" >"$state/$name.analysis.log"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_compact.py --manifest "$manifest" --compact "$state/$name.jsonl" --report "$state/$name.json" --validators "$result" --output "$state/$name.independent.json" >"$state/$name.independent.log"
done
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/aggregate_outcome_report.py >"$state/aggregate.log"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_outcome_raw_ids.py >"$state/raw_ids.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
rm -f "$state/suite.failed"
