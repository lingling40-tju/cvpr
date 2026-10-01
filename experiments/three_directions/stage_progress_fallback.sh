#!/usr/bin/env bash
set -euo pipefail

# Run only after the six-model branch evaluation has finished. This creates
# a small, separate source tree; it does not start any GPU service or trainer.
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
target="$base/ActiveVLN_progress_fallback_20261002"
full_eval="$source_root/runlogs/three_direction_scale_branch_128step_full_eval_all"
analysis="$source_root/runlogs/three_direction_full_val_unseen/scale_branch_128_analysis.json"
patch_file="$source_root/tools/geodesic_progress.patch"

test -f "$full_eval/suite.completed" || { echo 'complete branch evaluation is still running' >&2; exit 2; }
test -s "$analysis" && test -s "$patch_file"
test ! -e "$target" || { echo "target already exists: $target" >&2; exit 2; }
patch_sha=$(sha256sum "$patch_file" | awk '{print $1}')
[ "$patch_sha" = 982bd7df1d196acf7698abfe4768b589626c6122a207ec5b74c48b911939bc86 ] || {
  echo 'progress patch hash mismatch' >&2; exit 1;
}

check_source() {
  local relative=$1 expected=$2 actual
  actual=$(sha256sum "$source_root/$relative" | awk '{print $1}')
  [ "$actual" = "$expected" ] || { echo "source changed: $relative" >&2; exit 1; }
}
check_source vlnce_server/env_config.py d3f8dd8553d71bf61a830b19898df7a8c2d207ed3b614fbe11066e2c6c6bff47
check_source vlnce_server/semantic_reward/env.py d9cf6db6d0e098c707f2df261d3f6e39d36bf7a1f8d617934357e10814a6d057
check_source verl/workers/agent/parallel_env_vlnce.py 90d66792129a4e9fab29aa3ee247ceb86a6216197a668bbf03640c453107aa1e

mkdir -p "$target"
for dir in vlnce_server verl examples tools; do cp -a "$source_root/$dir" "$target/$dir"; done
ln -s "$source_root/data" "$target/data"
mkdir -p "$target/runlogs" "$target/verl_checkpoints" "$target/outputs"
patch --dry-run -d "$target" -p1 <"$patch_file"
patch -d "$target" -p1 <"$patch_file"
"$base/activevln_train_env/bin/python" -m py_compile \
  "$target/vlnce_server/env_config.py" \
  "$target/vlnce_server/semantic_reward/env.py" \
  "$target/vlnce_server/semantic_reward/progress.py" \
  "$target/verl/workers/agent/parallel_env_vlnce.py"
printf 'source=%s\npatch_sha256=%s\n' "$source_root" "$patch_sha" \
  >"$target/progress_source.txt"
echo "$target"
