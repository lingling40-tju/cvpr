#!/usr/bin/env bash
set -euo pipefail

# Reproduce compact publication inputs only after all six-model evaluations
# and all three paired training audits have completed.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
training="$root/runlogs/three_direction_scale_branch_128step_suite"
audits="$root/runlogs/three_direction_scale_branch_128step_pair_audit"
screen_suite="$root/runlogs/three_direction_scale_branch_128step_eval"
full_suite="$root/runlogs/three_direction_scale_branch_128step_full_eval_all"
screen="$root/runlogs/three_direction_val256"
full="$root/runlogs/three_direction_full_val_unseen"
output="$root/runlogs/three_direction_scale_branch_128step_publication"

for marker in "$training/suite.completed" "$audits/watcher.completed" \
              "$screen_suite/suite.completed" "$full_suite/suite.completed"; do
  test -f "$marker" || { echo "required stage incomplete: $marker" >&2; exit 2; }
done
for seed in 11 22 33; do test -s "$training/seed${seed}_pair_audit.json"; done
test -s "$screen/scale_branch_128_analysis.json"
test -s "$full/scale_branch_128_analysis.json"
[ "$(sha256sum "$screen/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46 ]
[ "$(sha256sum "$full/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e ]

mkdir -p "$output"
exec 9>"$output/package.lock"
flock -n 9 || { echo 'publication package already active' >&2; exit 2; }
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$output/package.failed"; fi
}
trap on_exit EXIT
rm -f "$output/package.failed"
cd "$root"
python="$base/activevln_server_env/bin/python"

for spec in "256:$screen" "1839:$full"; do
  count=${spec%%:*}
  result=${spec#*:}
  for seed in 11 22 33; do
    "$python" tools/export_matched_episodes.py \
      --root "$result" --candidate "branch128_seed${seed}" \
      --control "branch_control128_seed${seed}" \
      --expected-count "$count" \
      --output "$output/paired_${count}_seed${seed}.jsonl"
  done
done

"$python" tools/render_scaled_table.py \
  --screen "$screen/scale_branch_128_analysis.json" \
  --full "$full/scale_branch_128_analysis.json" \
  --output "$output/scaled_results_table.tex"
cp "$screen/scale_branch_128_analysis.json" "$output/val256_analysis.json"
cp "$full/scale_branch_128_analysis.json" "$output/full1839_analysis.json"
for seed in 11 22 33; do
  cp "$training/seed${seed}_pair_audit.json" "$output/seed${seed}_pair_audit.json"
done
(
  cd "$output"
  sha256sum paired_*.jsonl scaled_results_table.tex val256_analysis.json \
    full1839_analysis.json seed*_pair_audit.json >SHA256SUMS
)
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$output/package.completed"
