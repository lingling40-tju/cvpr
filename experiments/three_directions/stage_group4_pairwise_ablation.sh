#!/usr/bin/env bash
set -euo pipefail

# Stage the compute-matched GRPO grouping in a separate source tree. This
# command prepares code only; it never starts a simulator or trainer.
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
target="$base/ActiveVLN_group4_pairwise_20261002"
expected_trainer_sha=1d1334ac7c4267b32e6354bdc27a4e313b27dc025b6d6d1f535d5729b8420289
expected_patched_trainer_sha=2a2a623f9843eb234779bcede6495045a682df5e146bf6178c35531b7425cc9f
expected_patch_sha=e61a0e03b7df1c05d9bc99eee2272b88d4a962b46c4dec28b7f8ad598db6dfbb
expected_helper_sha=bff28117918d4f033896144d0585e814c50a609164422dd50ee67c648adfc825

hash_file() { sha256sum "$1" | awk '{print $1}'; }
test "$(hash_file "$source_root/verl/trainer/ppo/ray_trainer.py")" = "$expected_trainer_sha"
test "$(hash_file "$source_root/tools/group4_pairwise_uid.patch")" = "$expected_patch_sha"
test "$(hash_file "$source_root/tools/group4_pairwise_uid.py")" = "$expected_helper_sha"
if test -e "$target"; then
  test -s "$target/pairwise_source.txt"
  grep -Fx "source_trainer_sha256=$expected_trainer_sha" "$target/pairwise_source.txt" >/dev/null
  grep -Fx "patch_sha256=$expected_patch_sha" "$target/pairwise_source.txt" >/dev/null
  test "$(hash_file "$target/verl/trainer/ppo/group4_pairwise_uid.py")" = "$expected_helper_sha"
  test "$(hash_file "$target/verl/trainer/ppo/ray_trainer.py")" = "$expected_patched_trainer_sha"
  echo "$target"
  exit 0
fi

mkdir -p "$target"
for dir in vlnce_server verl examples tools; do cp -a "$source_root/$dir" "$target/$dir"; done
ln -s "$source_root/data" "$target/data"
mkdir -p "$target/runlogs" "$target/verl_checkpoints" "$target/outputs"
cp "$source_root/tools/group4_pairwise_uid.py" "$target/verl/trainer/ppo/group4_pairwise_uid.py"
patch --dry-run -d "$target" -p1 <"$source_root/tools/group4_pairwise_uid.patch"
patch -d "$target" -p1 <"$source_root/tools/group4_pairwise_uid.patch"
test "$(hash_file "$target/verl/trainer/ppo/ray_trainer.py")" = "$expected_patched_trainer_sha"
"$base/activevln_train_env/bin/python" -m py_compile \
  "$target/verl/trainer/ppo/ray_trainer.py" \
  "$target/verl/trainer/ppo/group4_pairwise_uid.py"
"$base/activevln_train_env/bin/python" "$target/tools/verify_group4_pairwise_uid.py"
printf 'source=%s\nsource_trainer_sha256=%s\npatch_sha256=%s\nhelper_sha256=%s\n' \
  "$source_root" "$expected_trainer_sha" "$expected_patch_sha" "$expected_helper_sha" \
  >"$target/pairwise_source.txt"
echo "$target"
