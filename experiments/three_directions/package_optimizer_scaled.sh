#!/usr/bin/env bash
set -euo pipefail

mode=${1:?group4 or kl_anchor required}
case "$mode" in group4|kl_anchor) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
gate="$root/runlogs/${mode}_scale_conditional"
eval_suite="$root/runlogs/optimizer_scale_eval_conditional"
screen="$root/runlogs/three_direction_val256"
full="$root/runlogs/three_direction_full_val_unseen"
output="$root/runlogs/${mode}_scale_publication"
test -f "$gate/training.completed"
test -f "$eval_suite/suite.completed"
test "$(cat "$gate/pilot_decision.txt")" = eligible
for seed in 11 22 33; do test -f "$gate/train_seed${seed}.completed"; done
mkdir -p "$output"
exec 9>"$output/package.lock"
flock -n 9 || { echo 'optimizer package already active' >&2; exit 2; }
if test -f "$output/package.completed"; then exit 0; fi
rm -f "$output/package.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$output/package.failed"; fi
}
trap on_exit EXIT
cd "$root"
python="$base/activevln_server_env/bin/python"

for spec in "256:$screen" "1839:$full"; do
  count=${spec%%:*}
  result=${spec#*:}
  if [ "$count" = 256 ]; then
    sha=546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46
    prefix=val256
  else
    sha=262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
    prefix=full1839
  fi
  test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = "$sha"
  test -s "$result/scale_${mode}_128_analysis.json"
  cp "$result/manifest.json" "$output/${prefix}_manifest.json"
  cp "$result/scale_${mode}_128_analysis.json" "$output/${prefix}_analysis.json"
  for seed in 11 22 33; do
    label="${mode}_128_seed${seed}"
    control="branch_control128_seed${seed}"
    test -f "$result/$label.completed"
    test -f "$result/$control.completed"
    test -s "$result/paired_${label}_vs_${control}.json"
    "$python" tools/export_matched_episodes.py \
      --root "$result" --candidate "$label" --control "$control" \
      --expected-count "$count" \
      --output "$output/paired_${count}_seed${seed}.jsonl"
  done
done

for seed in 11 22 33; do
  experiment="three_directions_${mode}_128step"
  if [ "$seed" != 11 ]; then experiment="${experiment}_seed${seed}"; fi
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
