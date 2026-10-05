#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_rloo_20261005"
control_root="$base/ActiveVLN_three_directions_20261002"
train="$root/runlogs/turn_rloo_64step_seed11"
result="$root/runlogs/turn_rloo_val_seen256"
mkdir -p "$result"
exec 9>"$result/suite.lock"
flock -n 9 || { echo 'evaluation watcher already active' >&2; exit 2; }
if test -f "$result/suite.completed"; then exit 0; fi
rm -f "$result/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$result/suite.failed"; fi; }
trap on_exit EXIT
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  d9ba66de3fb4fc070cae9449decb1427e80672d61523891af6f8d8ad45ad6a31

while ! test -f "$train/completed"; do
  test ! -f "$train/failed" || { echo 'training failed' >&2; exit 1; }
  test -s "$root/runlogs/turn_rloo_64.launcher.pid"
  kill -0 "$(cat "$root/runlogs/turn_rloo_64.launcher.pid")" 2>/dev/null || {
    echo 'training launcher vanished without completed marker' >&2; exit 1;
  }
  sleep 60
done

"$base/activevln_train_env/bin/python" - "$train/train.log" <<'PY' >"$result/train_audit.json"
import json,re,sys
lines=open(sys.argv[1],errors='replace')
steps={}
for line in lines:
    match=re.search(r'step:(\d+) - global_seqlen',line)
    if not match:continue
    number=int(match.group(1))
    grad=re.search(r'actor/grad_norm:([-+0-9.eE]+)',line)
    if not grad:raise ValueError(f'missing actor gradient at step {number}')
    steps[number]=float(grad.group(1))
if set(steps)!=set(range(1,65)) or any(x<=0 for x in steps.values()):
    raise ValueError(f'incomplete or zero-gradient training: {len(steps)} steps')
print(json.dumps({'steps':len(steps),'nonzero_gradient_steps':sum(x>0 for x in steps.values()),
                  'min_actor_grad_norm':min(steps.values())},indent=2))
PY

# Release the isolated training simulator before Habitat evaluation shards.
pid_file="$root/runlogs/service/server.pid"
if curl -fsS --max-time 2 http://127.0.0.1:5056/health >/dev/null 2>&1; then
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  ps -o args= -p "$pid" | grep -F 'server.port=5056' >/dev/null
  kill "$pid"
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5056/health >/dev/null 2>&1; then break; fi
    sleep 2
  done
  ! curl -fsS --max-time 1 http://127.0.0.1:5056/health >/dev/null 2>&1
fi

control_label=qwen3_exact_control_64step_seed11
candidate_label=turn_rloo_64step_seed11
control_model="$control_root/verl_checkpoints/$control_label/global_step_64/actor/huggingface"
candidate_model="$root/verl_checkpoints/$candidate_label/global_step_64/actor/huggingface"
test -f "$control_model/config.json" && test -f "$candidate_model/config.json"
export VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json"
export VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT=8125 VLN_VLLM_SEED=11
bash "$root/tools/run_val_seen_eval.sh" "$control_label" "$control_model" 0 2 \
  >"$result/$control_label.launcher.log" 2>&1
bash "$root/tools/run_val_seen_eval.sh" "$candidate_label" "$candidate_model" 0 2 \
  >"$result/$candidate_label.launcher.log" 2>&1
test -f "$result/$control_label.completed" && test -f "$result/$candidate_label.completed"
"$base/activevln_server_env/bin/python" "$root/tools/analyze_pair.py" \
  "$result" "$control_label" "$candidate_label" \
  --output "$result/paired_analysis.json" >"$result/analysis.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.completed"
