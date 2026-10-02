#!/usr/bin/env bash
set -euo pipefail

steps=${1:?2, 64, or 128 steps required}
seed=${2:?seed required}
case "$steps:$seed" in 2:11|64:11|128:11|128:22|128:33) ;; *) exit 2 ;; esac
pairwise_gpus=${VLN_PAIRWISE_GPUS:-2,3}
case "$pairwise_gpus" in 0,1|2,3) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_group4_pairwise_20261002"
experiment="three_directions_group4_pairwise_${steps}step_seed${seed}"
run_dir="$root/runlogs/$experiment"
checkpoint_dir="$root/verl_checkpoints/$experiment"
mkdir -p "$run_dir" "$checkpoint_dir"
exec 9>"$run_dir/run.lock"
flock -n 9 || { echo "pairwise run already active: $experiment" >&2; exit 2; }
if test -f "$run_dir/completed"; then
  test -s "$run_dir/paired_train_audit.json"
  test -f "$checkpoint_dir/global_step_${steps}/actor/huggingface/config.json"
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
export CUDA_VISIBLE_DEVICES="$pairwise_gpus"
export VLN_ALTERNATIVE_MODE=""
export VLN_GROUP4_PAIRWISE_ABLATION=1
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TENSORBOARD_DIR="$run_dir/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"

if [ "$steps" = 2 ] || [ "$steps" = 64 ]; then
  dataset=data/branch_pilot_train.parquet
  expected_sha=a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3
  source_run=three_directions_group4_64step_seed11
else
  dataset=data/branch_scale512_train.parquet
  expected_sha=2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea
  source_run=three_directions_group4_128step
  if [ "$seed" != 11 ]; then source_run="${source_run}_seed${seed}"; fi
fi
test -f "$source_root/runlogs/$source_run/completed"
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected_sha"
test "$(sha256sum verl/trainer/ppo/ray_trainer.py | awk '{print $1}')" = \
  2a2a623f9843eb234779bcede6495045a682df5e146bf6178c35531b7425cc9f
test "$(sha256sum verl/trainer/ppo/group4_pairwise_uid.py | awk '{print $1}')" = \
  bff28117918d4f033896144d0585e814c50a609164422dd50ee67c648adfc825
curl -fsS --max-time 5 http://127.0.0.1:5017/health >/dev/null
printf 'steps=%s seed=%s gpus=%s dataset=%s dataset_sha256=%s service=%s rollout_n=4 pairwise_uid=1 source_run=%s\n' \
  "$steps" "$seed" "$pairwise_gpus" "$dataset" "$expected_sha" http://127.0.0.1:5017 "$source_run" \
  >"$run_dir/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" \
  'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed="$seed" \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed="$seed" actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5017 \
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

python tools/audit_group4_pairwise_training.py --root "$root" \
  --source-root "$source_root" --steps "$steps" --seed "$seed" \
  >"$run_dir/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/completed"
