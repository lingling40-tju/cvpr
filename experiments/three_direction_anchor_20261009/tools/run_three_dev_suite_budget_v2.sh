#!/usr/bin/env bash
set -Eeuo pipefail
freeze_sha=${1:?evaluation identity SHA}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_direction_20261009"
sftroot="$base/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_development"
sftcheckpoint="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
state="$root/runlogs/development_suite"
result="$root/runlogs/development256"
mkdir -p "$state" "$result"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'development suite already active' >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
test ! -f "$state/suite.failed"
on_exit() { rc=$?; if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$state/suite.failed"; fi; }
trap on_exit EXIT
cd "$root"
check_freeze() {
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_three_eval_freeze.py --root "$root" --identity-sha "$freeze_sha"
}
check_freeze >"$state/freeze_before.json"
while test ! -f "$root/runlogs/sequential_training/training.completed"; do
  test ! -f "$root/runlogs/sequential_training/sequence.failed"
  owner=$(cat "$root/runlogs/sequential_training/sequence.launcher.pid")
  ps -o args= -p "$owner" | grep -F 'bash tools/run_three_direction_sequence.sh' >/dev/null
  sleep 30
done
exec 8>"$root/runlogs/sequential_training/sequence.lock"
flock 8
test ! -f "$root/runlogs/sequential_training/sequence.failed"
for method in grpo_anchor turn_rloo srgpo; do
  CUDA_VISIBLE_DEVICES="" "$base/activevln_train_env/bin/python" tools/audit_three_training_per_episode_budget.py --root "$root" --method "$method" --output "$state/${method}_train_audit.json" >"$state/${method}_train_audit.log"
done
check_freeze >"$state/freeze_after_training.json"
pidfile="$root/runlogs/service/server.pid"
if test -s "$pidfile"; then
  pid=$(cat "$pidfile")
  if ps -o args= -p "$pid" | grep -F 'server.port=5086' >/dev/null; then kill "$pid"; fi
fi
for _ in $(seq 1 90); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5086/health >/dev/null 2>&1; then break; fi
  sleep 2
done
if curl -fsS --max-time 1 http://127.0.0.1:5086/health >/dev/null 2>&1; then echo 'own Habitat service did not stop' >&2; exit 1; fi
wait_idle() {
  for _ in $(seq 1 180); do
    m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    m2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
    if test "$m0" -lt 10000 && test "$m1" -lt 10000 && test "$m2" -lt 10000; then return; fi
    sleep 10
  done
  echo "GPUs not idle: $m0 $m1 $m2" >&2; return 1
}
wait_idle
test -f "$sftroot/positive_initial_sft.completed"
test ! -f "$sftroot/positive_initial_sft.failed"
if test -L "$result/positive_initial_sft"; then
  test "$(readlink "$result/positive_initial_sft")" = "$sftroot/positive_initial_sft"
else
  test ! -e "$result/positive_initial_sft"
  ln -s "$sftroot/positive_initial_sft" "$result/positive_initial_sft"
fi
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest prepared_data/development256.json --role development --label positive_initial_sft --output "$result/positive_initial_sft.validated.json" >"$state/sft_reuse_validation.log"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py --log "$sftroot/vllm_positive_initial_sft.log" --checkpoint "$sftcheckpoint" --expected-dtype float16 --output "$result/positive_initial_sft.dtype.json" >"$state/sft_reuse_dtype.log"
check_freeze >"$state/freeze_before_inference.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/positive_initial_sft.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/development.opened"
bash tools/run_three_dev_model.sh grpo_anchor 0 "$freeze_sha" >"$state/grpo_anchor_eval.launcher.log" 2>&1 &
p0=$!
bash tools/run_three_dev_model.sh turn_rloo 1 "$freeze_sha" >"$state/turn_rloo_eval.launcher.log" 2>&1 &
p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
wait_idle
bash tools/run_three_dev_model.sh srgpo 0 "$freeze_sha" >"$state/srgpo_eval.launcher.log" 2>&1
for method in grpo_anchor turn_rloo srgpo; do
  label="td_${method}_n4_seed11"
  name="${method}_vs_sft"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py --root "$result" --manifest prepared_data/development256.json --role development --control positive_initial_sft --candidate "$label" --compact "$state/$name.jsonl" --output "$state/$name.json" >"$state/$name.analysis.log"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_compact.py --manifest prepared_data/development256.json --compact "$state/$name.jsonl" --report "$state/$name.json" --validators "$result" --output "$state/$name.independent.json" >"$state/$name.independent.log"
done
for method in turn_rloo srgpo; do
  name="${method}_vs_grpo"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py --root "$result" --manifest prepared_data/development256.json --role development --control td_grpo_anchor_n4_seed11 --candidate "td_${method}_n4_seed11" --compact "$state/$name.jsonl" --output "$state/$name.json" >"$state/$name.analysis.log"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_compact.py --manifest prepared_data/development256.json --compact "$state/$name.jsonl" --report "$state/$name.json" --validators "$result" --output "$state/$name.independent.json" >"$state/$name.independent.log"
done
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/aggregate_three_development.py --root "$root" --identity-sha "$freeze_sha" >"$state/aggregate.log"
check_freeze >"$state/freeze_after.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
