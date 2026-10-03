#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_turnwise_oracle_20261004"
test ! -e "$root"
check() {
  local path=$1 expected=$2
  test "$(sha256sum "$source_root/$path" | awk '{print $1}')" = "$expected"
}
check data/qwen3_group4_exact256.parquet \
  6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69
check vlnce_server/env.py \
  39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a
check verl/workers/agent/parallel_env_vlnce.py \
  90d66792129a4e9fab29aa3ee247ceb86a6216197a668bbf03640c453107aa1e
check verl/trainer/ppo/ray_trainer.py \
  1d1334ac7c4267b32e6354bdc27a4e313b27dc025b6d6d1f535d5729b8420289
check verl/trainer/ppo/core_algos.py \
  e135f843794ebaf781ee07088eea97e4039c4a3dfaeb58773ca84b96f61b5c2e

# Source and small data only; the large existing checkpoints remain in place.
mkdir -p "$root"
cp -a "$source_root"/{data,eval,examples,tools,verl,vlnce_server} "$root/"
mkdir -p "$root"/{runlogs,verl_checkpoints,outputs}
printf 'source=%s\ndataset_sha256=%s\nenv_sha256=%s\nagent_sha256=%s\ntrainer_sha256=%s\ncore_algos_sha256=%s\n' \
  "$source_root" \
  6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69 \
  39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a \
  90d66792129a4e9fab29aa3ee247ceb86a6216197a668bbf03640c453107aa1e \
  1d1334ac7c4267b32e6354bdc27a4e313b27dc025b6d6d1f535d5729b8420289 \
  e135f843794ebaf781ee07088eea97e4039c4a3dfaeb58773ca84b96f61b5c2e \
  >"$root/runlogs/tree_provenance.txt"
echo "prepared $root"
