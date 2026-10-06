#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_norm_terminal_rloo_n8_20261006"
original="$base/ActiveVLN_norm_terminal_rloo_20261006"
gae="$base/ActiveVLN_turn_gae_20261006/runlogs/turn_gae_val_seen778"
state="$root/runlogs/posthoc_n8_pilot"
result="$original/runlogs/norm_terminal_val_seen256"
mkdir -p "$state"
exec 9>"$state/watcher.lock"
flock -n 9 || { echo 'n8 pilot watcher already active' >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
rm -f "$state/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi; }
trap on_exit EXIT
while ! test -f "$gae/suite.verified"; do
  test ! -f "$gae/suite.failed" || { echo 'GAE evaluation failed; diagnose before sharing GPUs' >&2; exit 1; }
  test -s "$gae/suite.launcher.pid"
  pid=$(cat "$gae/suite.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'GAE evaluation watcher disappeared' >&2; exit 1; }
  sleep 60
done
test ! -f "$gae/suite.failed"
test "$(sha256sum "$original/runlogs/norm_terminal_val_seen256/manifest.json" | awk '{print $1}')" = 39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b
test -f "$original/runlogs/norm_terminal_val_seen256/suite.completed"
test "$(sha256sum "$root/verl/trainer/ppo/normalized_terminal_rloo.py" | awk '{print $1}')" = 9de66ec17fceb2615c58961f3d30f2dd681e14bb3f7675f5c69b9b0869ceff55
test "$(sha256sum "$root/verl/trainer/ppo/turn_rloo_advantage.py" | awk '{print $1}')" = ece4813ed3998650520bd161e5ad83ea7bc1991041aaaadd5787450a9d311876
"$base/activevln_train_env/bin/python" "$root/verl/trainer/ppo/normalized_terminal_rloo.py" >"$state/self_check.log"
for attempt in $(seq 1 360); do
  busy=0
  for port in 5062 5075 8132 8133 8134; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then busy=1; fi
  done
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$busy" -eq 0 && test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 60
done
test "$busy" -eq 0 && test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000
bash "$root/tools/start_service.sh" >"$state/service_start.log" 2>&1
bash "$root/tools/run_train.sh" 2 >"$state/smoke.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" "$original/tools/audit_training_gradients.py" "$root/runlogs/norm_terminal_n8_2step_seed11/train.log" 2 "$state/smoke_gradients.json" >"$state/smoke_audit.log"
bash "$root/tools/run_train.sh" 64 >"$state/train.launcher.log" 2>&1
"$base/activevln_train_env/bin/python" "$original/tools/audit_training_gradients.py" "$root/runlogs/norm_terminal_n8_64step_seed11/train.log" 64 "$state/train_gradients.json" >"$state/train_audit.log"
pidfile="$root/runlogs/service/server.pid"
test -s "$pidfile"
pid=$(cat "$pidfile")
ps -o args= -p "$pid" | grep -F 'server.port=5062' >/dev/null
kill "$pid"
for attempt in $(seq 1 90); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5062/health >/dev/null 2>&1; then break; fi
  sleep 2
done
! curl -fsS --max-time 1 http://127.0.0.1:5062/health >/dev/null 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"
label=norm_terminal_n8_64step_seed11
checkpoint="$root/verl_checkpoints/$label/global_step_64/actor/huggingface"
test -f "$checkpoint/config.json"
VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT=8134 VLN_VLLM_SEED=11 \
  bash "$base/ActiveVLN_turn_rloo_terminal_20261006/tools/run_val_seen_eval.sh" "$label" "$checkpoint" 0 2 >"$state/eval.launcher.log" 2>&1
test -f "$result/$label.completed"
"$base/activevln_server_env/bin/python" "$original/tools/analyze_pair.py" "$result" qwen3_exact_control_64step_seed11 "$label" --output "$state/n8_vs_control.json" >"$state/analyze_control.log"
"$base/activevln_server_env/bin/python" "$original/tools/analyze_pair.py" "$result" norm_terminal_rloo_64step_seed11 "$label" --output "$state/n8_vs_n4.json" >"$state/analyze_n4.log"
"$base/activevln_server_env/bin/python" "$root/tools/posthoc_verify_n8.py" "$result" "$label" "$state/n8_vs_control.json" "$state/n8_vs_n4.json" >"$state/independent_recount.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
