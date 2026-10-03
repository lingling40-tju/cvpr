#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/group_relative_collection"
output="$root/runlogs/ordinal_progress/group_relative_turns"
manifest="$root/runlogs/ordinal_progress/group_relative_manifest.json"
expected_sha=a99a15020b3d7ffe061ea82ab830e4d617e5ad8f90e3fff8979ab80079a26356
old_manifest="$root/runlogs/ordinal_progress/policy_process_manifest.json"
old_root="$root/runlogs/ordinal_progress/policy_process_turns"
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'group-relative replay already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_sha"
test "$(sha256sum "$old_manifest" | awk '{print $1}')" = \
  aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_server_env/bin/python"
collect() {
  local part=$1 shard=$2 shards=$3 gpu=$4
  "$python" tools/collect_policy_process_turns.py \
    --manifest "$manifest" --part "$part" --output-root "$output" \
    --gpu "$gpu" --shard "$shard" --shards "$shards" \
    --reuse-manifest "$old_manifest" --reuse-root "$old_root" \
    >"$run/${part}_${shard}.log" 2>&1
}
collect fit 0 2 0 & pid0=$!
collect fit 1 2 1 & pid1=$!
collect development 0 1 2 & pid2=$!
collect audit 0 1 3 & pid3=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
wait "$pid3" || status=1
test "$status" -eq 0
for part in fit development audit; do
  shards=1
  if test "$part" = fit; then shards=2; fi
  "$python" tools/merge_policy_process_shards.py \
    --manifest "$manifest" --output-root "$output" \
    --part "$part" --shards "$shards" \
    >"$run/${part}_merge.log" 2>&1
done
"$python" tools/audit_policy_process_turns.py \
  --manifest "$manifest" --output-root "$output" \
  --audit-output "$run/collection_audit.json" \
  >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
