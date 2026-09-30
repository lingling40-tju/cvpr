#!/usr/bin/env bash
set -euo pipefail

# Run one matched R2R train experiment on wanghaozhihuoshanyun.
# Usage: bash tools/run_multiseed_train.sh 11 control|event
seed=${1:?seed required}
arm=${2:?control or event required}
case "$arm" in
  control) semantic_weight=0 ;;
  event) semantic_weight=1 ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
train_env="$base/activevln_train_env"
experiment="eventtrace_r2r64_seed${seed}_${arm}"
checkpoint_dir="$root/verl_checkpoints/$experiment"

cd "$root"
export PATH="$train_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1,2
export RAY_TMPDIR="/tmp/eventtrace_r2r64_seed${seed}_${arm}"
export RAY_ADDRESS=local
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TENSORBOARD_DIR="$root/runlogs/tensorboard/$experiment"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR" "$checkpoint_dir"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  'data.train_files=[data/r2r_4000_train.parquet]' \
  'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=true "+data.seed=$seed" \
  data.train_batch_size=4 \
  data.val_batch_size=4 \
  data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  "+actor_rollout_ref.rollout.seed=$seed" \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5002 \
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
  actor_rollout_ref.rollout.agent.reward.semantic_reward_weight="$semantic_weight" \
  actor_rollout_ref.rollout.agent.reward.semantic_verifier_url=http://127.0.0.1:5003 \
  actor_rollout_ref.rollout.agent.reward.semantic_event_manifest="" \
  trainer.n_gpus_per_node=2 \
  trainer.total_training_steps=64 \
  trainer.save_freq=32 \
  trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' \
  trainer.resume_mode=disable \
  trainer.project_name=activevln \
  trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint_dir"
