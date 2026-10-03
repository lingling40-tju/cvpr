#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_qwen_confident_20261004"
root="$base/ActiveVLN_stop_pair_group4_20261004"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$root"
test -f "$source_root/runlogs/qwen_confident_64step_seed11/completed"
test "$(sha256sum "$source_root/verl/workers/agent/parallel_env_vlnce.py" | awk '{print $1}')" = \
  c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256
test "$(sha256sum "$source_root/vlnce_server/semantic_reward/env.py" | awk '{print $1}')" = \
  d2420040cede386967f2203df48af16cf1dd2cac7a9732b662071eab0021eabd
test "$(sha256sum "$source_root/data/qwen3_group4_exact256.parquet" | awk '{print $1}')" = \
  6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69
test "$(sha256sum "$script_dir/stop_pair_group4_reward.py" | awk '{print $1}')" = \
  1a075580917155633959ff4d4fb572cbe01a194b4957fc7a45d1be35e6a6c89c
test "$(sha256sum "$script_dir/apply_stop_pair_agent.py" | awk '{print $1}')" = \
  7d3a3bc1ad58f7d624a093cbe158aad249a8042a68f2d073aeb81ffd4bed0de0

# Copy source and small frozen train data; leave checkpoints and logs in place.
mkdir -p "$root"
cp -a "$source_root"/{data,eval,examples,tools,verl,vlnce_server} "$root/"
mkdir -p "$root"/{runlogs,verl_checkpoints,outputs}
cp "$script_dir/stop_pair_group4_reward.py" \
  "$root/verl/workers/agent/stop_pair_group4_reward.py"
cp "$script_dir/stop_pair_group4_reward.py" \
  "$root/tools/stop_pair_group4_reward.py"
"$base/activevln_server_env/bin/python" "$script_dir/apply_stop_pair_agent.py" \
  --root "$root"
printf 'source=%s\nsource_agent_sha256=%s\nstop_pair_helper_sha256=%s\n' \
  "$source_root" \
  c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256 \
  1a075580917155633959ff4d4fb572cbe01a194b4957fc7a45d1be35e6a6c89c \
  >"$root/runlogs/tree_provenance.txt"
echo "prepared isolated STOP-pair source tree: $root"
