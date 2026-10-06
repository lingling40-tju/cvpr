#!/usr/bin/env bash
set -euo pipefail
steps=${1:?pass 2 or 64}
case "$steps" in 2|64) ;; *) echo 'only 2 or 64 steps allowed' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
test -f "$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/conditional_chain/chain.completed"
test -f "$root/runlogs/conditional_chain/normalized_failed_gate.completed"
experiment="turn_gae_${steps}step_seed11"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "GAE run already active: $experiment" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi; }
trap on_exit EXIT
cd "$root"
dataset=data/qwen3_group4_exact256.parquet
test "$(sha256sum "$dataset" | awk '{print $1}')" = 6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69
test "$(sha256sum verl/utils/model.py | awk '{print $1}')" = 5e3bff8da3fd5dd9c1889314a537cc853a3a74af05c3509521fad650df544d0e
test "$(sha256sum verl/trainer/main_ppo.py | awk '{print $1}')" = 1a1be33b15a395c6799f17272eb23a14ff8416fb7f394a31c6aeefcfa69c0c1f
test "$(sha256sum verl/trainer/ppo/ray_trainer.py | awk '{print $1}')" = d0e782c68617400517ddb0404f88aa73d50475c57e6019fc2daa9b4661be5e0b
curl -fsS --max-time 5 http://127.0.0.1:5075/health >"$run/habitat_health_before.json"
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,3 RAY_ADDRESS=local
unset VLN_ORACLE_TURNWISE VLN_TURN_RLOO VLN_NORMALIZED_TERMINAL_RLOO
export VLN_ALTERNATIVE_MODE=""
export RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'seed=11 steps=%s dataset_sha256=%s group_n=4 method=multimodal_gae gamma=0.99 lambda=0.95 reward=terminal_outcome actor_gpus=0,1 habitat_gpu=2 critic_gpu=3\n' \
  "$steps" 6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69 >"$run/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" 'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed=11 \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed=11 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5075 \
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
  algorithm.adv_estimator=gae algorithm.gamma=0.99 algorithm.lam=0.95 \
  critic.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size_per_gpu=1 \
  critic.optim.lr=1e-5 \
  trainer.n_gpus_per_node=3 +trainer.critic_gpus_per_node=1 \
  trainer.total_training_steps="$steps" trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1

test -f "$checkpoint/global_step_${steps}/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
