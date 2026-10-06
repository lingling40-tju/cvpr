#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
terminal="$base/ActiveVLN_turn_rloo_terminal_20261006"
normalized="$base/ActiveVLN_norm_terminal_rloo_20261006"
control="$base/ActiveVLN_three_directions_20261002"
factorial="$terminal/runlogs/factorial_val_seen256"
chain="$terminal/runlogs/factorial_chain"
state="$normalized/runlogs/conditional_chain"
result="$normalized/runlogs/norm_terminal_val_seen256"
mkdir -p "$state" "$result"
exec 9>"$state/chain.lock"
flock -n 9 || { echo 'normalized conditional watcher already active' >&2; exit 2; }
if test -f "$state/chain.completed"; then exit 0; fi
rm -f "$state/chain.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$state/chain.failed"; fi; }
trap on_exit EXIT

# Wait for the existing factorial trainer/evaluator to release the GPUs.
while ! test -f "$chain/chain.completed"; do
  test ! -f "$chain/chain.failed" && test ! -f "$factorial/suite.failed" || {
    echo 'factorial chain failed; conditional decision requires recovery' >&2; exit 1;
  }
  test -s "$chain/chain.launcher.pid"
  pid=$(cat "$chain/chain.launcher.pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    test -f "$chain/chain.completed" || { echo 'factorial watcher disappeared' >&2; exit 1; }
  fi
  sleep 60
done
test -f "$factorial/suite.completed"

# Independently recount every arm before deciding whether to spend more GPU.
"$base/activevln_server_env/bin/python" "$terminal/tools/export_factorial_compact.py" \
  "$factorial" --output "$factorial/factorial_compact.json"
"$base/activevln_server_env/bin/python" "$terminal/tools/verify_factorial_compact.py" \
  "$factorial/factorial_compact.json" "$factorial/manifest.json" \
  --output "$factorial/factorial_independent_recount.json" \
  >"$state/factorial_independent_recount.log"
if python3 - "$factorial" "$state" <<'PY'
import json,sys
from pathlib import Path
root,state=map(Path,sys.argv[1:])
remote=json.loads((root/'factorial_decision.json').read_text())
ind=json.loads((root/'factorial_independent_recount.json').read_text())
assert remote['manifest_sha256']==ind['manifest_sha256']=='c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325'
assert ind['episodes']==256 and ind['scenes']==50 and ind['inference_errors']==0
assert set(remote['comparisons'])==set(ind['comparisons'])
for name,row in ind['comparisons'].items():
    other=remote['comparisons'][name]
    assert abs(row['paired_sr_points']-other['paired_sr_points'])<1e-6
    assert abs(row['paired_spl_points']-other['paired_spl_points'])<1e-6
    assert row['advance_gate_passed']==other['advance_gate_passed']
passed=[name for name,row in ind['comparisons'].items() if row['advance_gate_passed']]
(state/'factorial_gate.json').write_text(json.dumps({'passed':passed,'independent_recount':str(root/'factorial_independent_recount.json')},indent=2)+'\n')
print('passing candidates:',passed)
sys.exit(0 if passed else 10)
PY
then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/skipped_for_scale.completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/chain.completed"
  exit 0
else
  # Exit 10 means the independent recount found no passing arm.
  # A Python exception exits 1 and must not launch another experiment.
  status=$?
  test "$status" -eq 10
fi

for port in 5057 5058 5059; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "Habitat service still active on $port" >&2; exit 1
  fi
done
for attempt in $(seq 1 60); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000

bash "$normalized/tools/start_service.sh"
bash "$normalized/tools/run_train.sh" 2
python3 "$terminal/tools/audit_training_gradients.py" \
  "$normalized/runlogs/norm_terminal_rloo_2step_seed11/train.log" 2 \
  "$state/norm_2_train_audit.json"
bash "$normalized/tools/run_train.sh" 64
python3 "$terminal/tools/audit_training_gradients.py" \
  "$normalized/runlogs/norm_terminal_rloo_64step_seed11/train.log" 64 \
  "$state/norm_64_train_audit.json"

pidfile="$normalized/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5059' >/dev/null
kill "$pid"
for attempt in $(seq 1 60); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5059/health >/dev/null 2>&1

manifest_source="$terminal/runlogs/normalized_terminal_contingency/manifest.json"
manifest="$result/manifest.json"
if test ! -f "$manifest"; then cp "$manifest_source" "$manifest"; fi
test "$(sha256sum "$manifest" | awk '{print $1}')" = 39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b
export VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$manifest"
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
run_label norm_terminal_rloo_64step_seed11 \
  "$normalized/verl_checkpoints/norm_terminal_rloo_64step_seed11/global_step_64/actor/huggingface" 1 8127 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
test "$status" -eq 0
"$base/activevln_server_env/bin/python" "$terminal/tools/analyze_pair.py" \
  "$result" qwen3_exact_control_64step_seed11 norm_terminal_rloo_64step_seed11 \
  --output "$result/paired_norm_terminal_rloo_64step_seed11.json" \
  >"$result/analysis.log"
python3 - "$result" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
x=json.loads((root/'paired_norm_terminal_rloo_64step_seed11.json').read_text())
assert x['episodes']==256 and x['scenes']==38
assert (root/'qwen3_exact_control_64step_seed11.completed').exists()
assert (root/'norm_terminal_rloo_64step_seed11.completed').exists()
out={'schema':'norm_terminal_rloo_conditional_decision_v1',
     'manifest_sha256':'39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b',
     'episodes':256,'scenes':38,
     'paired_sr_points':x['paired_sr_points'],
     'paired_spl_points':x['paired_spl_points'],
     'advance_gate_passed':x['paired_sr_points']>=2 and x['paired_spl_points']>=2,
     'interpretation':'One-seed val-seen development screen only; not unseen-scene generalization.'}
(root/'conditional_decision.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/chain.completed"
