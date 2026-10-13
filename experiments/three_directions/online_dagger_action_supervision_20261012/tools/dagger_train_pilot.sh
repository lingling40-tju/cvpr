#!/usr/bin/env bash
set -euo pipefail
arm=${1:?control or candidate}
case "$arm" in control|candidate) ;; *) exit 2 ;; esac
steps=64
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_online_dagger_action_supervision_20261012"
cd "$root"
experiment="online_dagger_${arm}_64step_seed11"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "training already active: $experiment" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
test ! -f "$run/failed" || { echo "previous failed marker exists: $run/failed" >&2; exit 2; }
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s
' "$status" >"$run/failed"; fi; }
trap on_exit EXIT
dataset=prepared_data/fit512.parquet
expected=240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected"
curl -fsS --max-time 5 http://127.0.0.1:5087/health >"$run/habitat_health_before.json"
test -s "$root/runlogs/dagger_service/server.pid"
service_pid=$(cat "$root/runlogs/dagger_service/server.pid")
ps -o args= -p "$service_pid" | grep -F 'server.port=5087' >/dev/null
test "$(cat "$root/runlogs/dagger_service/mode")" = zero
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1 VLN_ORACLE_TURNWISE=1 VLN_TURN_RLOO=1
export VLN_NORMALIZED_TERMINAL_RLOO=0 VLN_POSITIVE_TRAJECTORY=0 VLN_ALTERNATIVE_MODE=""
export VLN_TURN_RLOO_PROGRESS_WEIGHT=0.0 VLN_TURN_REWARD_MODE=zero
export VLN_ORACLE_ACTION_MATCH=0 VLN_DAGGER_TEACHER=1
export RAY_ADDRESS=local RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
if test "$arm" = candidate; then dagger_coef=0.05; else dagger_coef=0.0; fi
printf 'method=on_policy_shortest_path_dagger_ce arm=%s seed=11 steps=%s fit_rows=512 batch=8 group_n=4 dataset_sha256=%s terminal_reward=weighted_success15_plus_ndtw5 rloo=turn_level process_reward=zero dagger_ce_coef=%s teacher_targets_per_update=32 init=SFT_fresh
'   "$arm" "$steps" "$expected" "$dagger_coef" >"$run/config.txt"
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo   --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml   "data.train_files=[$dataset]" 'data.val_files=[data/r2r_val_tiny.parquet]'   data.shuffle=false +data.seed=11 data.train_batch_size=8 data.val_batch_size=8 data.max_response_length=8192   actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"   +actor_rollout_ref.rollout.seed=11 actor_rollout_ref.rollout.n=4   actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1   actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean   actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.01   actor_rollout_ref.actor.dagger_ce_coef="$dagger_coef" algorithm.use_kl_in_reward=false   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2   actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5087   actor_rollout_ref.rollout.agent.timeout=300 actor_rollout_ref.rollout.agent.max_turn_budget=12   actor_rollout_ref.rollout.agent.max_step_budget=36   actor_rollout_ref.rollout.agent.single_response_max_tokens=256   actor_rollout_ref.rollout.agent.max_vllm_images=32   actor_rollout_ref.rollout.agent.enable_dynamic_sampling=false   actor_rollout_ref.rollout.agent.enable_fallback=false   actor_rollout_ref.rollout.agent.prob_from_scrath=1   actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw   actor_rollout_ref.rollout.agent.reward.success_reward_base=15   actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=5   actor_rollout_ref.rollout.agent.reward.semantic_success_floor=0   actor_rollout_ref.rollout.agent.reward.semantic_reward_weight=0   trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps"   trainer.save_freq="$steps" trainer.test_freq=-1 'trainer.logger=[console,tensorboard]'   trainer.resume_mode=disable trainer.project_name=activevln   trainer.experiment_name="$experiment" trainer.default_local_dir="$checkpoint"   >"$run/train.log" 2>&1
test -f "$checkpoint/global_step_2/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
