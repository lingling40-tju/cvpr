#!/usr/bin/env bash
set -euo pipefail

steps=${1:?pass 2 or 64}
case "$steps" in 2|64) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_qwen_group_rank_20261004"
control="$base/ActiveVLN_three_directions_20261002"
experiment="qwen_group_rank_${steps}step_seed11"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'candidate already running' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test -f "$control/runlogs/qwen3_exact_control_64step_seed11/completed"
cd "$root"
dataset=data/qwen3_group4_exact256.parquet
expected=6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected"
test "$(sha256sum verl/workers/agent/parallel_env_vlnce.py | awk '{print $1}')" = \
  c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256
test "$(sha256sum vlnce_server/semantic_reward/env.py | awk '{print $1}')" = \
  d2420040cede386967f2203df48af16cf1dd2cac7a9732b662071eab0021eabd
curl -fsS --max-time 5 http://127.0.0.1:5031/health >"$run/habitat_health_before.json"
curl -fsS --max-time 5 http://127.0.0.1:8031/health >"$run/reward_health_before.json"
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2,3
export VLN_ALTERNATIVE_MODE=""
export RAY_ADDRESS=local
export RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'seed=11 steps=%s dataset_sha256=%s rollout_n=4 reward=qwen_group_same_mode_ord_half service=5031 teacher=8031\n' \
  "$steps" "$expected" >"$run/config.txt"

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
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5031 \
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
  +actor_rollout_ref.rollout.agent.reward.fused_reward_url=http://127.0.0.1:8031 \
  +actor_rollout_ref.rollout.agent.reward.qwen_group_relative=true \
  trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps" \
  trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1

curl -fsS --max-time 5 http://127.0.0.1:8031/health >"$run/reward_health_after.json"
test -f "$checkpoint/global_step_${steps}/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
