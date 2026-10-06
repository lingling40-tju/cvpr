#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
evaluator="$base/ActiveVLN_turn_rloo_terminal_20261006"
control="$base/ActiveVLN_three_directions_20261002/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
candidate="$root/verl_checkpoints/turn_gae_64step_seed11/global_step_64/actor/huggingface"
train="$root/runlogs/turn_gae_64step_seed11"
result="$root/runlogs/turn_gae_val_seen778"
mkdir -p "$result"
exec 9>"$result/suite.lock"
flock -n 9 || { echo 'GAE evaluation suite already active' >&2; exit 2; }
if test -f "$result/suite.verified"; then exit 0; fi
if test -f "$result/suite.completed"; then
  echo 'unverified suite completion marker exists; inspect before rerun' >&2
  exit 1
fi
rm -f "$result/suite.failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then
    rm -f "$result/suite.completed"
    printf '%s\n' "$status" >"$result/suite.failed"
  fi
}
trap on_exit EXIT

while ! test -f "$train/completed"; do
  test ! -f "$train/failed" || { echo 'GAE training failed' >&2; exit 1; }
  test -s "$root/runlogs/conditional_chain/gae_64.launcher.pid"
  pid=$(cat "$root/runlogs/conditional_chain/gae_64.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'GAE trainer disappeared' >&2; exit 1; }
  sleep 60
done
test ! -f "$train/failed"
test "$(sha256sum "$root/tools/audit_gae_gradients.py" | awk '{print $1}')" = 0e318d842fa3af1abd080f7a621952412f161b4f043b674caae9fd5564ac5849
"$base/activevln_train_env/bin/python" "$root/tools/audit_gae_gradients.py" \
  "$train/train.log" 64 "$result/train_64_gradient_audit.json" \
  >"$result/train_64_audit.log"
test -f "$control/config.json" && test -f "$candidate/config.json"
test "$(sha256sum "$evaluator/tools/run_val_seen_eval.sh" | awk '{print $1}')" = 11df0b1811cd965bfb083c71914da4b840ffe508020235f1417a615ffc3dcb58
test "$(sha256sum "$evaluator/tools/eval_val_seen_subset.py" | awk '{print $1}')" = fec2cd3c0ff8912319dacca5b5616d5b6b850d96cced926980a0bc33037607b5
test "$(sha256sum "$root/tools/verify_gae_raw.py" | awk '{print $1}')" = 968461039bf760606d2051a3a378746aa4682376ca2804ff48fc26910e140319
test "$(sha256sum "$root/tools/preverify_gae_suite.py" | awk '{print $1}')" = 0fc82ab795cf4de43715e6b3ce07639c5707a866d73fa02fba82a2cc003551bc
if test ! -f "$result/manifest.json"; then
  cp "$root/tools/val_seen778_manifest.json" "$result/manifest.json"
fi
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = 03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed

pidfile="$root/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5075' >/dev/null
kill "$pid"
for attempt in $(seq 1 60); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5075/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5075/health >/dev/null 2>&1
for attempt in $(seq 1 60); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem3=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000 && test "$mem3" -lt 10000

run_arm() {
  local label=$1 model=$2 gpu=$3 port=$4
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=778 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" VLN_VLLM_SEED=11 \
    bash "$evaluator/tools/run_val_seen_eval.sh" "$label" "$model" "$gpu" 2 \
    >"$result/$label.launcher.log" 2>&1
  test -f "$result/$label.completed"
}
run_arm qwen3_exact_control_64step_seed11 "$control" 0 8132 &
first=$!
run_arm turn_gae_64step_seed11 "$candidate" 1 8133 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
test "$status" -eq 0

# First independent raw recount is completed before any suite marker.
"$base/activevln_server_env/bin/python" "$root/tools/preverify_gae_suite.py" \
  "$result" "$result/manifest.json" "$result/precompletion_recount.json" \
  >"$result/precompletion_recount.log"
test -s "$result/precompletion_recount.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.completed"

# A second implementation re-reads every raw episode and must agree.
"$base/activevln_server_env/bin/python" "$root/tools/verify_gae_raw.py" \
  "$result" "$result/manifest.json" \
  --compact "$result/compact.json" \
  --output "$result/independent_recount.json" \
  >"$result/independent_recount.log"
python3 - "$result" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
pre=json.loads((root/'precompletion_recount.json').read_text())
post=json.loads((root/'independent_recount.json').read_text())
assert pre['manifest_sha256']==post['manifest_sha256']=='03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed'
assert pre['episodes']==post['episodes']==778 and pre['scenes']==post['scenes']==53
assert pre['inference_errors']==post['inference_errors']==0
for key in ('control_successes','candidate_successes','candidate_only_success','control_only_success'):
    assert pre[key]==post[key],key
for key in ('control_spl','candidate_spl','paired_sr_points','paired_spl_points'):
    assert abs(pre[key]-post[key])<1e-6,key
assert pre['advance_gate_passed']==post['advance_gate_passed']
(root/'verified_gate.json').write_text(json.dumps({'advance_gate_passed':post['advance_gate_passed'],
  'paired_sr_points':post['paired_sr_points'],'paired_spl_points':post['paired_spl_points'],
  'precompletion_raw_stats_sha256':pre['raw_stats_sha256']},indent=2)+'\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.verified"
