#!/usr/bin/env bash
set -euo pipefail

mode=${1:?branch, recovery, counterfactual, or matched control}
steps=${2:-64}
gpus=${3:-0,1}
seed=${4:-11}
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
case "$mode" in
  branch|recovery|counterfactual)
    dataset="data/${mode}_pilot_train.parquet"
    alternative_mode="$mode"
    ;;
  branch_control|recovery_control|counterfactual_control)
    dataset="data/${mode%_control}_pilot_train.parquet"
    alternative_mode=""
    ;;
  *) echo "unknown direction: $mode" >&2; exit 2 ;;
esac
dataset=${VLN_TRAIN_DATASET:-$dataset}

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
train_env="$base/activevln_train_env"
service_url=${VLN_PILOT_SERVICE_URL:-http://127.0.0.1:5002}
experiment="three_directions_${mode}_${steps}step"
if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
checkpoint_dir="$root/verl_checkpoints/$experiment"
run_dir="$root/runlogs/$experiment"
mkdir -p "$checkpoint_dir" "$run_dir"
rm -f "$run_dir/failed" "$run_dir/completed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/failed"; fi
}
trap on_exit EXIT
cd "$root"

export PATH="$train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpus"
export VLN_ALTERNATIVE_MODE="$alternative_mode"
ray_tag=$(printf '%s' "$experiment" | cksum | awk '{print $1}')
export RAY_TMPDIR="/tmp/td_${ray_tag}"
export RAY_ADDRESS=local
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TENSORBOARD_DIR="$run_dir/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
test -f "$dataset"
curl -fsS --max-time 5 "$service_url/health" >/dev/null
printf 'mode=%s steps=%s gpus=%s seed=%s dataset=%s service=%s\n' "$mode" "$steps" "$gpus" "$seed" "$dataset" "$service_url" >"$run_dir/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" \
  'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed="$seed" \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed="$seed" actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url="$service_url" \
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
  trainer.n_gpus_per_node=2 trainer.total_training_steps="$steps" \
  trainer.save_freq="$steps" trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint_dir" \
  >"$run_dir/train.log" 2>&1

if [ "$mode" = counterfactual ]; then
  python3 tools/check_counterfactual_run.py "$checkpoint_dir/rollout.jsonl" \
    "$run_dir/train.log" --steps "$steps" >"$run_dir/validation.json"
else
  python3 tools/check_grpo_preflight.py "$checkpoint_dir/rollout.jsonl" \
    "$run_dir/train.log" --steps "$steps" >"$run_dir/validation.json"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/completed"
