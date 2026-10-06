#!/usr/bin/env bash
set -euo pipefail
seed=${1:?pass seed 11, 22, or 33}
case "$seed" in 11|22|33) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_norm_terminal_rloo_20261006"
gae_state="$base/ActiveVLN_turn_gae_20261006/runlogs/conditional_chain"
pilot="$root/runlogs/norm_terminal_val_seen256"
test -f "$pilot/suite.completed"
test ! -f "$pilot/suite.failed"
test -f "$gae_state/skipped_for_scale.completed"
test -f "$gae_state/watcher.completed"
python3 - "$gae_state/normalized_gate.json" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text())
assert x['normalized_manifest_sha256'] == '39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b'
assert x['advance_gate_passed'] is True and x['decision'] == 'skip_gae_for_scale'
assert x['paired_sr_points'] >= 2.0 and x['paired_spl_points'] >= 2.0
PY
experiment="norm_terminal_rloo_exact512_128_seed${seed}"
run="$root/runlogs/$experiment"
checkpoint="$root/verl_checkpoints/$experiment"
mkdir -p "$run" "$checkpoint"
exec 9>"$run/run.lock"
flock -n 9 || { echo "scale training already active: $experiment" >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi; }
trap on_exit EXIT
cd "$root"
dataset=data/qwen3_group4_exact512.parquet
expected=d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f
test "$(sha256sum "$dataset" | awk '{print $1}')" = "$expected"
test "$(sha256sum verl/trainer/ppo/turn_rloo_advantage.py | awk '{print $1}')" = ece4813ed3998650520bd161e5ad83ea7bc1991041aaaadd5787450a9d311876
test "$(sha256sum verl/trainer/ppo/normalized_terminal_rloo.py | awk '{print $1}')" = 772d00a438d32a716b2c666d681e40af8794375d5e075b381194c2006cf80fa1
curl -fsS --max-time 5 http://127.0.0.1:5059/health >"$run/habitat_health_before.json"
export PATH="$base/activevln_train_env/bin:$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1 VLN_ORACLE_TURNWISE=1 VLN_TURN_RLOO=1 VLN_NORMALIZED_TERMINAL_RLOO=1
export VLN_ALTERNATIVE_MODE="" RAY_ADDRESS=local
export RAY_TMPDIR="/dev/shm/td_$(printf '%s' "$experiment" | cksum | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export TENSORBOARD_DIR="$run/tensorboard"
mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
printf 'seed=%s steps=128 dataset_sha256=%s group_n=4 method=normalized_terminal_rloo outcome_scale=15 progress_weight=0.0 gamma=1 source=terminal_outcome_only token_advantage=per_turn_standardized\n' \
  "$seed" "$expected" >"$run/config.txt"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-path "$root/examples/vlnce" --config-name train_vlnce_4gpus.yaml \
  "data.train_files=[$dataset]" 'data.val_files=[data/r2r_val_tiny.parquet]' \
  data.shuffle=false +data.seed="$seed" \
  data.train_batch_size=4 data.val_batch_size=4 data.max_response_length=8192 \
  actor_rollout_ref.model.path="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  +actor_rollout_ref.rollout.seed="$seed" actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.agent.base_url=http://127.0.0.1:5059 \
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
  trainer.save_freq=64 trainer.test_freq=-1 \
  'trainer.logger=[console,tensorboard]' trainer.resume_mode=auto \
  trainer.project_name=activevln trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$checkpoint" >"$run/train.log" 2>&1

test -f "$checkpoint/global_step_128/actor/huggingface/config.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
