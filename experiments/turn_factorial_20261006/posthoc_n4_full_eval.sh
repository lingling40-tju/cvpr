#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_norm_terminal_rloo_20261006"
control="$base/ActiveVLN_three_directions_20261002"
control_result="$base/ActiveVLN_turnwise_oracle_20261004/runlogs/oracle_exact512_full1839"
scale="$root/runlogs/posthoc_n4_scale"
state="$root/runlogs/posthoc_n4_full1839"
result="$state/results"
mkdir -p "$state" "$result"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'normalized scale full evaluation already active' >&2; exit 2; }
if test -f "$state/watcher.completed"; then exit 0; fi
rm -f "$state/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi; }
trap on_exit EXIT

while ! test -f "$scale/scale_train.completed"; do
  test ! -f "$scale/watcher.failed" || { echo 'posthoc scale training failed' >&2; exit 1; }
  test -s "$scale/watcher.launcher.pid"
  pid=$(cat "$scale/watcher.launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo 'posthoc scale watcher disappeared' >&2; exit 1; }
  sleep 60
done
test -f "$scale/watcher.completed"
test ! -f "$scale/watcher.failed"
test -f "$root/runlogs/norm_terminal_val_seen256/suite.completed"
test "$(sha256sum "$control/tools/run_direction_eval.sh" | awk '{print $1}')" = 82c985013efd9ba76f25768ed704a266f7f90694dddffe711e99faceadc922af
test "$(sha256sum "$control/tools/eval_val_unseen_subset.py" | awk '{print $1}')" = 555c66b62b23dd815ebca91df5f0cbef7876eef1eabfe1c250c2ef0510f23c4e
test "$(sha256sum "$root/tools/verify_three_seed_scale_raw.py" | awk '{print $1}')" = f2ff880ff35d955afc4f789b9a9cd354aafbdc856fb1e09c91d62d102cbd6fc0
if test ! -f "$result/manifest.json"; then
  cp "$control_result/manifest.json" "$result/manifest.json"
fi
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = 262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
test "$(sha256sum "$control_result/manifest.json" | awk '{print $1}')" = 262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
for seed in 11 22 33; do
  label="oracle_exact512_control_128_seed${seed}"
  test -f "$control_result/$label.completed"
  test ! -f "$control_result/$label.failed"
  test -f "$control/verl_checkpoints/$label/global_step_128/actor/huggingface/config.json"
  test -f "$root/runlogs/norm_terminal_posthoc512_128_seed${seed}/completed"
  test -f "$root/verl_checkpoints/norm_terminal_posthoc512_128_seed${seed}/global_step_128/actor/huggingface/config.json"
done
for port in 5059 8130 8131; do
  ! curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1
  ! curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1
done
for attempt in $(seq 1 60); do
  mem0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000; then break; fi
  sleep 10
done
test "$mem0" -lt 10000 && test "$mem1" -lt 10000 && test "$mem2" -lt 10000

evaluate_seed() {
  local seed=$1 gpu=$2 port=$3 label checkpoint
  label="norm_terminal_posthoc512_128_seed${seed}"
  checkpoint="$root/verl_checkpoints/$label/global_step_128/actor/huggingface"
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" VLN_VLLM_SEED="$seed" \
    bash "$control/tools/run_direction_eval.sh" "$label" "$checkpoint" "$gpu" 2 \
    >"$state/$label.launcher.log" 2>&1
  test -f "$result/$label.completed"
}
evaluate_seed 11 0 8130 &
first=$!
evaluate_seed 22 1 8131 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
test "$status" -eq 0
evaluate_seed 33 0 8130

"$base/activevln_server_env/bin/python" "$root/tools/verify_three_seed_scale_raw.py" \
  --candidate-root "$result" \
  --candidate-pattern 'norm_terminal_posthoc512_128_seed{seed}' \
  --control-root "$control_result" \
  --control-pattern 'oracle_exact512_control_128_seed{seed}' \
  --manifest "$result/manifest.json" \
  --compact-dir "$state/compact" \
  --output "$state/independent_three_seed_recount.json" \
  >"$state/independent_recount.log"
test -s "$state/independent_three_seed_recount.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/watcher.completed"
