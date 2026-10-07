#!/usr/bin/env bash
set -euo pipefail
arm=${1:?pass control or candidate}
seed=${2:?pass seed 11, 22, or 33}
case "$arm" in control) gpu=0; port=8136 ;; candidate) gpu=1; port=8137 ;; *) exit 2 ;; esac
case "$seed" in 11|22|33) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/positive_scale"
label="positive_trajectory_${arm}_128step_seed${seed}"
checkpoint="$root/verl_checkpoints/$label/global_step_128/actor/huggingface"
result="$root/runlogs/positive_reserved256"
manifest="$root/prepared_data/reserved256.json"
# Refuse inference until all six matched training audits have passed.
test -f "$state/training.completed"
test ! -f "$state/suite.failed"
"$base/activevln_server_env/bin/python" - "$state" "$root/runlogs/positive_pilot/frozen_gate.json" <<'PY'
from pathlib import Path
import json, sys
state = Path(sys.argv[1])
g = json.load(open(sys.argv[2]))
assert g["schema"] == "positive_trajectory_frozen_development_gate_v1"
assert g["pass"] is True and g["paired_sr_points"] >= 2 and g["paired_spl_points"] >= 2
assert g["reserved_screen_opened"] is False
for seed in (11, 22, 33):
    for arm in ("control", "candidate"):
        a = json.loads((state / f"{arm}_seed{seed}_train_audit.json").read_text())
        assert a["arm"] == arm and a["expected_steps"] == a["observed_steps"] == 128
        assert a["positive_advantage_steps"] > 0 and a["nonzero_actor_gradient_steps"] > 0
        assert a["max_terminal_score"] <= 20.001
        assert a["kl_loss_metric_present_and_finite"] is True
        if arm == "candidate": assert a["min_advantage"] >= -0.001
PY
mkdir -p "$result" "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu${gpu}.lock"
flock 8
exec 9>"$result/$label.lock"
flock 9
test -f "$root/runlogs/$label/completed"
test -f "$checkpoint/config.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = 412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241
if test -f "$result/$label.completed"; then exit 0; fi
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "port $port is occupied" >&2
  exit 1
fi
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
test "$memory" -lt 10000
if curl -fsS --max-time 1 http://127.0.0.1:5085/health >/dev/null 2>&1; then
  echo 'Training Habitat service still running' >&2
  exit 1
fi
rm -f "$result/$label.failed"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
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
    --manifest "$manifest" --result-root "$result" --role reserved \
    --count 256 --shard-count 4 --shard-index "$shard" --max-turns 12 \
    --base-url "http://127.0.0.1:$port/v1" \
    >"$result/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
"$base/activevln_server_env/bin/python" tools/validate_train_label.py \
  --root "$result" --manifest "$manifest" --role reserved \
  --label "$label" --output "$result/$label.validated.json" \
  >"$result/$label.validation.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/$label.completed"
