#!/usr/bin/env bash
set -euo pipefail

steps=${1:?pass 2 or 64}
case "$steps" in 2|64) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_mode_rank_20261003"
source_root="$base/ActiveVLN_three_directions_20261002"
experiment="mode_rank_group4_${steps}step_seed11"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'mode-rank pilot already running' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
test -f "$source_root/runlogs/three_directions_group4_${steps}step_seed11/completed"
test "$(sha256sum "$root/verl/workers/agent/parallel_env_vlnce.py" | awk '{print $1}')" = \
  5a35ed8f48337b8362a44e6ef31e5e4ed546c481372aba8c6ab6d72433c094fa
cd "$root"
dataset=data/branch_pilot_train.parquet
expected_sha=a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected_sha"
curl -fsS --max-time 3 http://127.0.0.1:5026/health >"$run/habitat_health_before.json"
curl -fsS --max-time 3 http://127.0.0.1:8026/health >"$run/reward_health_before.json"
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2,3
export VLN_ALTERNATIVE_MODE=""
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'steps=%s seed=11 dataset_sha256=%s group_size=4 ordinal_mode_stratified=true frozen_failure_reward_url=8026 simulator=5026\n' \
  "$steps" "$expected_sha" >"$run/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" \
  'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed=11 \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed=11 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5026 \
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
  +actor_rollout_ref.rollout.agent.reward.fused_reward_weight=1 \
  +actor_rollout_ref.rollout.agent.reward.fused_reward_url=http://127.0.0.1:8026 \
  +actor_rollout_ref.rollout.agent.reward.mode_stratified_ordinal=true \
  trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps" \
  trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" \
  >"$run/train.log" 2>&1

curl -fsS --max-time 3 http://127.0.0.1:8026/health >"$run/reward_health_after.json"
test -f "$checkpoint/global_step_${steps}/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
