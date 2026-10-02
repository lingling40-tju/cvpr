#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_group4_pairwise_20261002"
source_root="$base/ActiveVLN_three_directions_20261002"
gate="$root/runlogs/group4_pairwise_scale_conditional"
eval_suite="$root/runlogs/group4_pairwise_scale_eval_conditional"
output="$root/runlogs/group4_pairwise_scale_publication"
test "$(cat "$gate/pilot_decision.txt")" = eligible
test -f "$gate/training.completed"
test -f "$eval_suite/suite.completed"
for seed in 11 22 33; do test -f "$gate/train_seed${seed}.completed"; done
mkdir -p "$output"
exec 9>"$output/package.lock"
flock -n 9 || { echo 'pairwise scale package already active' >&2; exit 2; }
if test -f "$output/package.completed"; then exit 0; fi
rm -f "$output/package.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$output/package.failed"; fi
}
trap on_exit EXIT

for count in 256 1839; do
  if [ "$count" = 256 ]; then
    result="$source_root/runlogs/three_direction_val256"
    expected_sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
    prefix=val256
  else
    result="$source_root/runlogs/three_direction_full_val_unseen"
    expected_sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
    prefix=full1839
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$expected_sha"
  test -s "$result/scale_group4_pairwise_128_analysis.json"
  cp "$result/manifest.json" "$output/${prefix}_manifest.json"
  cp "$result/scale_group4_pairwise_128_analysis.json" "$output/${prefix}_analysis.json"
  for seed in 11 22 33; do
    candidate="group4_pairwise_128_seed${seed}"
    test -f "$result/$candidate.completed"
    for comparison in branch_control group4; do
      if [ "$comparison" = branch_control ]; then
        control="branch_control128_seed${seed}"
      else
        control="group4_128_seed${seed}"
      fi
      test -f "$result/$control.completed"
      test -s "$result/paired_${candidate}_vs_${control}.json"
      "$base/activevln_server_env/bin/python" "$source_root/tools/export_matched_episodes.py" \
        --root "$result" --candidate "$candidate" --control "$control" \
        --expected-count "$count" \
        --output "$output/paired_${count}_${comparison}_seed${seed}.jsonl"
    done
  done
done

for seed in 11 22 33; do
  experiment="three_directions_group4_pairwise_128step_seed${seed}"
  test -f "$root/runlogs/$experiment/completed"
  cp "$root/runlogs/$experiment/paired_train_audit.json" \
    "$output/seed${seed}_train_audit.json"
done
(
  cd "$output"
  sha256sum paired_*.jsonl *_analysis.json *_manifest.json \
    seed*_train_audit.json >SHA256SUMS
)
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$output/package.completed"
