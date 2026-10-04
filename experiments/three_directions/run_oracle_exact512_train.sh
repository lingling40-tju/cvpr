#!/usr/bin/env bash
set -euo pipefail

arm=${1:?control or candidate required}
seed=${2:?seed 11, 22, or 33 required}
case "$arm:$seed" in
  control:11|control:22|control:33|candidate:11|candidate:22|candidate:33) ;;
  *) exit 2 ;;
esac
base=/Knowin/foundation/haozhiwang/whz
control_root="$base/ActiveVLN_three_directions_20261002"
oracle_root="$base/ActiveVLN_turnwise_oracle_20261004"
test -f "$oracle_root/runlogs/oracle_exact512_scale/eligible"
if test "$arm" = candidate; then
  root="$oracle_root"
  experiment="oracle_turnwise_exact512_128_seed${seed}"
  port=5035
  turnwise=1
else
  root="$control_root"
  experiment="oracle_exact512_control_128_seed${seed}"
  port=5013
  turnwise=0
fi
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "scale training already active: $experiment" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
cd "$root"
dataset=data/qwen3_group4_exact512.parquet
expected=d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected"
curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$run/habitat_health_before.json"
if test "$arm" = candidate; then
  test "$(sha256sum verl/workers/agent/parallel_env_vlnce.py | awk '{print $1}')" = \
    903c3788902a613c68d8d4c0f681eacbe3d0d4eb421c7ea4d2a45f4cde648391
  test "$(sha256sum verl/trainer/ppo/ray_trainer.py | awk '{print $1}')" = \
    2a135d65f35d0a5d9108404746ff102e69afac171b9ce6dc6764fd9c3bf10393
  test "$(sha256sum verl/trainer/ppo/turnwise_group4_advantage.py | awk '{print $1}')" = \
    092e7917e1756bcbbdb56ec7a58efde970c8e4c6f59a0124f7f660133ed9c9a6
fi
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2,3
if test "$arm" = candidate; then
  export VLN_ORACLE_TURNWISE=1
else
  unset VLN_ORACLE_TURNWISE
fi
export VLN_ALTERNATIVE_MODE="" RAY_ADDRESS=local
export RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'arm=%s seed=%s steps=128 dataset_sha256=%s rollout_n=4 habitat_port=%s oracle_turnwise=%s\n' \
  "$arm" "$seed" "$expected" "$port" "$turnwise" >"$run/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" 'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed="$seed" \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed="$seed" actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url="http://127.0.0.1:$port" \
  actor_rollout_ref.rollout.agent.timeout=300 \
  actor_rollout_ref.rollout.agent.max_turn_budget=12 \
  actor_rollout_ref.rollout.agent.max_step_budget=36 \
  actor_rollout_ref.rollout.agent.single_response_max_tokens=256 \
  actor_rollout_ref.rollout.agent.max_vllm_images=32 \
  actor_rollout_ref.rollout.agent.enable_dynamic_sampling=false \
  actor_rollout_ref.rollout.agent.prob_from_scrath=1 \
  actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw \
  actor_rollout_ref.rollout.agent.reward.success_reward_base=15 \
  actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=0 \
  actor_rollout_ref.rollout.agent.reward.semantic_success_floor=2 \
  actor_rollout_ref.rollout.agent.reward.semantic_reward_weight=0 \
  trainer.n_gpus_per_node=2 trainer.total_training_steps=128 \
  trainer.save_freq=64 trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1

test -f "$checkpoint/global_step_128/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
