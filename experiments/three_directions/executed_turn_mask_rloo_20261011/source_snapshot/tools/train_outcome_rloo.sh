#!/usr/bin/env bash
set -euo pipefail
steps=${1:?2 smoke or 64 total updates}
gpu=${2:-0,1}
case "$steps" in 2|64) ;; *) exit 2 ;; esac
case "$gpu" in 0,1|0,3|1,3) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_executed_turn_rloo_20261011"
label=td_executed_turn_rloo_n4_seed11
run="$root/runlogs/$label"
checkpoint="$root/verl_checkpoints/$label"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "training already active: $label" >&2; exit 2; }
if test "$steps" = 2 && test -f "$run/smoke.completed"; then exit 0; fi
if test "$steps" = 64 && test -f "$run/completed"; then exit 0; fi
test ! -f "$run/failed"
on_exit() { rc=$?; if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/failed"; fi; }
trap on_exit EXIT
cd "$root"
test "$(sha256sum prepared_data/fit512.parquet | awk '{print $1}')" = 240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e
test "$(sha256sum prepared_data/development256.json | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
curl -fsS --max-time 5 http://127.0.0.1:5087/health >"$run/habitat_health_before.json"
test -s "$root/runlogs/service/server.pid"
ps -o args= -p "$(cat "$root/runlogs/service/server.pid")" | grep -F "server.port=5087" >/dev/null
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu" RAY_ADDRESS=local
export VLN_ORACLE_TURNWISE=1 VLN_TURN_RLOO=1 VLN_NORMALIZED_TERMINAL_RLOO=0 VLN_SRGPO=0
export VLN_TURN_RLOO_PROGRESS_WEIGHT=0.5 VLN_TURN_RLOO_GAMMA=1.0
export VLN_OUTCOME_CONSISTENT_STOP=1 VLN_NDTW_SUCCESS_ONLY=1
export VLN_POSITIVE_TRAJECTORY=0 VLN_ALTERNATIVE_MODE=""
export RAY_TMPDIR="/dev/shm/outcome_rloo_${gpu}_seed11"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'method=outcome_consistent_rloo gpu=%s requested_steps=%s total_updates=64 train_rows=512 batch_rows=8 group_n=4 configured_seed=11 kl_coef=0.1 lr=1e-6 forced_stop=eval_equivalent success_only_ndtw=1 progress_weight=0.5\n' \
  "$gpu" "$steps" >"$run/config.txt"
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  'data.train_files=[prepared_data/fit512.parquet]' 'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed=11 data.train_batch_size=8 data.val_batch_size=8 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed=11 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.1 algorithm.use_kl_in_reward=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5087 \
  actor_rollout_ref.rollout.agent.timeout=300 actor_rollout_ref.rollout.agent.max_turn_budget=12 \
  actor_rollout_ref.rollout.agent.max_step_budget=36 actor_rollout_ref.rollout.agent.single_response_max_tokens=256 \
  actor_rollout_ref.rollout.agent.max_vllm_images=32 actor_rollout_ref.rollout.agent.enable_dynamic_sampling=false \
  actor_rollout_ref.rollout.agent.prob_from_scrath=1 \
  actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw \
  actor_rollout_ref.rollout.agent.reward.success_reward_base=15 actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=5 \
  actor_rollout_ref.rollout.agent.reward.semantic_success_floor=0 actor_rollout_ref.rollout.agent.reward.semantic_reward_weight=0 \
  trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps" trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln_outcome_consistent trainer.experiment_name="$label" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1
if test "$steps" = 2; then
  test -f "$checkpoint/global_step_2/actor/huggingface/config.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/smoke.completed"
else
  test -f "$checkpoint/global_step_64/actor/huggingface/config.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
fi
