#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
scratch="$base/policy_action_memory_lora_20261004"
run="$scratch/runlogs/diversity_fit1024"
manifest="$scratch/diversity_fit1024_manifest.json"
old_manifest="$root/runlogs/ordinal_progress/policy_process_manifest.json"
output="$scratch/diversity_fit1024_turns"
old_output="$root/runlogs/ordinal_progress/policy_process_turns"
mkdir -p "$run" "$output"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'diversity fit replay already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37
test "$(sha256sum "$old_manifest" | awk '{print $1}')" = \
  aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
test -f "$root/runlogs/ordinal_progress/policy_process_collection/completed"
used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
  sed -n '2p' | tr -d ' ')
test -n "$used1" && test "$used1" -lt 8000
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_server_env/bin/python"
"$python" tools/collect_policy_process_turns.py \
  --manifest "$manifest" --part fit --output-root "$output" --gpu 1 \
  --reuse-root "$old_output" --reuse-manifest "$old_manifest" \
  >"$run/collect.log" 2>&1
"$python" "$scratch/audit_diversity_policy_fit.py" \
  --manifest "$manifest" --old-manifest "$old_manifest" \
  --records-root "$output" --old-records-root "$old_output" \
  --output "$run/fit_audit.json" >"$run/audit.log" 2>&1
"$python" - "$run/fit_audit.json" "$run" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1]));run=pathlib.Path(sys.argv[2])
assert x['schema']=='diversity_policy_fit_replay_audit_v1'
name='passed_sample_gate' if x['extra_sample_gate']['passed'] else 'failed_sample_gate'
(run/name).write_text('fit-only extra replay label coverage\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
