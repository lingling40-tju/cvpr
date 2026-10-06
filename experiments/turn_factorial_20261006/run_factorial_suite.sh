#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
terminal="$base/ActiveVLN_turn_rloo_terminal_20261006"
dense="$base/ActiveVLN_dense_grpo_20261006"
combined="$base/ActiveVLN_turn_rloo_20261005"
control="$base/ActiveVLN_three_directions_20261002"
result="$terminal/runlogs/factorial_val_seen256"
mkdir -p "$result"
exec 9>"$result/suite.lock"
flock -n 9 || { echo "factorial evaluator already active" >&2; exit 2; }
if test -f "$result/suite.completed"; then exit 0; fi
rm -f "$result/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$result/suite.failed"; fi; }
trap on_exit EXIT
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325
test -f "$terminal/runlogs/turn_rloo_terminal_64step_seed11/completed"
test -f "$dense/runlogs/dense_grpo_64step_seed11/completed"
test -f "$combined/runlogs/turn_rloo_64step_seed11/completed"
for port in 5057 5058; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "training simulator still active on port $port" >&2
    exit 1
  fi
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.started"
# Wait for FSDP and Ray to release the two inference GPUs.
for attempt in $(seq 1 30); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000
export VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json"
export VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_VLLM_SEED=11
run_label() {
  local label="$1" model="$2" gpu="$3" port="$4"
  test -f "$model/config.json"
  VLN_EVAL_PORT="$port" bash "$terminal/tools/run_val_seen_eval.sh" "$label" "$model" "$gpu" 2 \
    >"$result/$label.launcher.log" 2>&1
}
run_label qwen3_exact_control_64step_seed11 \
  "$control/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface" 0 8126 &
first=$!
run_label turn_rloo_64step_seed11 \
  "$combined/verl_checkpoints/turn_rloo_64step_seed11/global_step_64/actor/huggingface" 1 8127 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
test "$status" -eq 0
run_label turn_rloo_terminal_64step_seed11 \
  "$terminal/verl_checkpoints/turn_rloo_terminal_64step_seed11/global_step_64/actor/huggingface" 0 8126 &
first=$!
run_label dense_grpo_64step_seed11 \
  "$dense/verl_checkpoints/dense_grpo_64step_seed11/global_step_64/actor/huggingface" 1 8127 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
test "$status" -eq 0
for candidate in turn_rloo_64step_seed11 turn_rloo_terminal_64step_seed11 dense_grpo_64step_seed11; do
  "$base/activevln_server_env/bin/python" "$terminal/tools/analyze_pair.py" \
    "$result" qwen3_exact_control_64step_seed11 "$candidate" \
    --output "$result/paired_$candidate.json" >"$result/analysis_$candidate.log"
done
python3 - "$result" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
names=("turn_rloo_64step_seed11","turn_rloo_terminal_64step_seed11","dense_grpo_64step_seed11")
rows={}
for name in names:
    x=json.loads((root/f"paired_{name}.json").read_text())
    assert x["episodes"]==256 and x["scenes"]==50
    assert (root/f"{name}.completed").exists()
    rows[name]={
        "paired_sr_points":x["paired_sr_points"],
        "paired_spl_points":x["paired_spl_points"],
        "advance_gate_passed":x["paired_sr_points"]>=2 and x["paired_spl_points"]>=2,
    }
out={"schema":"terminal_credit_reward_factorial_decision_v1",
     "manifest_sha256":"c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325",
     "episodes":256,"scenes":50,"comparisons":rows,
     "interpretation":"One-seed val-seen development screen; no unseen-scene or semantic-reward claim."}
(root/"factorial_decision.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.completed"
