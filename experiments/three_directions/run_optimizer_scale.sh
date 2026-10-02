#!/usr/bin/env bash
set -euo pipefail

mode=${1:?group4 or kl_anchor required}
seed=${2:?seed 11, 22, or 33 required}
case "$mode" in group4|kl_anchor) ;; *) exit 2 ;; esac
case "$seed" in 11|22|33) ;; *) exit 2 ;; esac

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
dataset=data/branch_scale512_train.parquet
expected_sha=2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea
experiment="three_directions_${mode}_128step"
if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
checkpoint_dir="$root/verl_checkpoints/$experiment"
run_dir="$root/runlogs/$experiment"
mkdir -p "$checkpoint_dir" "$run_dir"
exec 9>"$run_dir/run.lock"
flock -n 9 || { echo "scale run already active: $experiment" >&2; exit 2; }
if test -f "$run_dir/completed"; then
  test -s "$run_dir/paired_train_audit.json"
  test -f "$checkpoint_dir/global_step_128/actor/huggingface/config.json"
  exit 0
fi
rm -f "$run_dir/failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run_dir/failed"; fi
}
trap on_exit EXIT
cd "$root"

export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export VLN_ALTERNATIVE_MODE=""
export RAY_TMPDIR="/tmp/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export RAY_ADDRESS=local
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TENSORBOARD_DIR="$run_dir/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
actual_sha=$(sha256sum "$dataset" | awk '{print $1}')
test "$actual_sha" = "$expected_sha" || { echo 'scale dataset hash mismatch' >&2; exit 1; }
if [ "$mode" = group4 ]; then
  gpus=2,3
  rollout_n=4
  service_url=http://127.0.0.1:5013
  kl_args=()
else
  gpus=0,1
  rollout_n=2
  service_url=http://127.0.0.1:5014
  kl_args=(actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 algorithm.kl_ctrl.kl_coef=0)
fi
export CUDA_VISIBLE_DEVICES="$gpus"
curl -fsS --max-time 5 "$service_url/health" >/dev/null
printf 'mode=%s steps=128 gpus=%s seed=%s dataset=%s dataset_sha256=%s service=%s rollout_n=%s actor_kl_coef=%s\n' \
  "$mode" "$gpus" "$seed" "$dataset" "$actual_sha" "$service_url" "$rollout_n" \
  "$([ "$mode" = kl_anchor ] && printf 0.001 || printf 0)" >"$run_dir/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" \
  'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed="$seed" \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed="$seed" actor_rollout_ref.rollout.n="$rollout_n" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  "${kl_args[@]}" \
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
  trainer.n_gpus_per_node=2 trainer.total_training_steps=128 \
  trainer.save_freq=128 trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint_dir" \
  >"$run_dir/train.log" 2>&1

python3 tools/audit_optimizer_scaled_pair.py --root "$root" --mode "$mode" --seed "$seed" \
  >"$run_dir/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/completed"
