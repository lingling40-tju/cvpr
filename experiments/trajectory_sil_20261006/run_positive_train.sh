#!/usr/bin/env bash
set -euo pipefail
arm=${1:?pass control or candidate}
steps=${2:?pass 2 or 64}
case "$arm" in control|candidate) ;; *) exit 2 ;; esac
case "$steps" in 2|64) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
experiment="positive_trajectory_${arm}_${steps}step_seed11"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "training already active: $experiment" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi; }
trap on_exit EXIT
cd "$root"
dataset=prepared_data/fit512.parquet
expected=240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected"
test "$(sha256sum verl/trainer/ppo/ray_trainer.py | awk '{print $1}')" = 50356c80a1fda653e10f4332d128b611d29a4268598023d37c66513c26f31cdb
test "$(sha256sum verl/trainer/ppo/positive_trajectory_advantage.py | awk '{print $1}')" = d7516263c565a276797b8f6b7d9d5d3976ae81f2d5ea47331870e8404f9ead69
curl -fsS --max-time 5 http://127.0.0.1:5085/health >"$run/habitat_health_before.json"
test -s "$root/runlogs/service/server.pid"
ps -o args= -p "$(cat "$root/runlogs/service/server.pid")" | grep -F 'server.port=5085' >/dev/null
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1 VLN_ORACLE_TURNWISE=0 VLN_TURN_RLOO=0 VLN_NORMALIZED_TERMINAL_RLOO=0
if test "$arm" = candidate; then export VLN_POSITIVE_TRAJECTORY=1; else export VLN_POSITIVE_TRAJECTORY=0; fi
export VLN_ALTERNATIVE_MODE="" RAY_ADDRESS=local
export RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'arm=%s seed=11 steps=%s rows=512 batch_rows=8 group_n=4 dataset_sha256=%s reward=weighted_success15_plus_ndtw5 loss_agg=seq-mean-token-mean kl_loss_coef=0.01\n' \
  "$arm" "$steps" "$expected" >"$run/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" 'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed=11 \
  data.train_batch_size=8 data.val_batch_size=8 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed=11 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.01 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5085 \
  actor_rollout_ref.rollout.agent.timeout=300 \
  actor_rollout_ref.rollout.agent.max_turn_budget=12 \
  actor_rollout_ref.rollout.agent.max_step_budget=36 \
  actor_rollout_ref.rollout.agent.single_response_max_tokens=256 \
  actor_rollout_ref.rollout.agent.max_vllm_images=32 \
  actor_rollout_ref.rollout.agent.enable_dynamic_sampling=false \
  actor_rollout_ref.rollout.agent.prob_from_scrath=1 \
  actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw \
  actor_rollout_ref.rollout.agent.reward.success_reward_base=15 \
  actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=5 \
  actor_rollout_ref.rollout.agent.reward.semantic_success_floor=2 \
  actor_rollout_ref.rollout.agent.reward.semantic_reward_weight=0 \
  trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps" \
  trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1

test -f "$checkpoint/global_step_${steps}/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
