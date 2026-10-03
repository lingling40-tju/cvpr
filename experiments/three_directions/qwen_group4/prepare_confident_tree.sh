#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_qwen_group_rank_20261004"
root="$base/ActiveVLN_qwen_confident_20261004"
pilot="$source_root/runlogs/qwen_group_followup"
scale="$source_root/runlogs/qwen_group_scale"
test -f "$pilot/completed"
test -f "$scale/no_pilot_gain"
test -f "$scale/suite.completed"
test ! -e "$root"
test "$(sha256sum "$source_root/verl/workers/agent/qwen_group_reward.py" | awk '{print $1}')" = \
  b915ff15b6d778df719b09e79719f2ba9714b3d873c3b373795e89c3e6ad6e1a
test "$(sha256sum "$source_root/verl/workers/agent/parallel_env_vlnce.py" | awk '{print $1}')" = \
  c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256
test "$(sha256sum "$source_root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  d2420040cede386967f2203df48af16cf1dd2cac7a9732b662071eab0021eabd
test "$(sha256sum "$source_root/tools/confident_pair_reward.py" | awk '{print $1}')" = \
  ad327090b281b3e6dd103badcf7173a86ead18ffd904f6bf6c85c097249690d9
test "$(sha256sum "$source_root/tools/next_val256_manifest.json" | awk '{print $1}')" = \
  e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1

# Code and small Parquet files are copied; 58-GB checkpoint/log trees stay put.
mkdir -p "$root"
cp -a "$source_root"/{data,eval,examples,tools,verl,vlnce_server} "$root/"
mkdir -p "$root"/{runlogs,verl_checkpoints,outputs}
cp "$source_root/tools/confident_pair_reward.py" \
  "$root/verl/workers/agent/qwen_group_reward.py"
cp "$source_root/tools/confident_pair_reward.py" \
  "$root/tools/qwen_group_reward.py"
test "$(sha256sum "$root/verl/workers/agent/qwen_group_reward.py" | awk '{print $1}')" = \
  ad327090b281b3e6dd103badcf7173a86ead18ffd904f6bf6c85c097249690d9
test "$(sha256sum "$root/data/qwen3_group4_exact256.parquet" | awk '{print $1}')" = \
  6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69
printf 'source=%s\noriginal_reward_sha256=%s\nconfident_reward_sha256=%s\n' \
  "$source_root" \
  b915ff15b6d778df719b09e79719f2ba9714b3d873c3b373795e89c3e6ad6e1a \
  ad327090b281b3e6dd103badcf7173a86ead18ffd904f6bf6c85c097249690d9 \
  >"$root/runlogs/tree_provenance.txt"
echo "prepared isolated confident-pair source tree: $root"
