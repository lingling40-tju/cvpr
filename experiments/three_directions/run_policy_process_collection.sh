#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_process_collection"
manifest="$root/runlogs/ordinal_progress/policy_process_manifest.json"
output="$root/runlogs/ordinal_progress/policy_process_turns"
expected_sha=aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'policy-process collection already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_sha"
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
  local part=$1 gpu=$2
  "$python" tools/collect_policy_process_turns.py \
    --manifest "$manifest" --part "$part" \
    --output-root "$output" --gpu "$gpu" \
    >"$run/$part.log" 2>&1
}

collect fit 0 & fit_pid=$!; printf '%s\n' "$fit_pid" >"$run/fit.pid"
collect development 1 & development_pid=$!; printf '%s\n' "$development_pid" >"$run/development.pid"
collect audit 2 & audit_pid=$!; printf '%s\n' "$audit_pid" >"$run/audit.pid"
status=0
wait "$fit_pid" || status=1
wait "$development_pid" || status=1
wait "$audit_pid" || status=1
test "$status" -eq 0
"$python" tools/audit_policy_process_turns.py \
  --manifest "$manifest" --output-root "$output" \
  --audit-output "$run/collection_audit.json" >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
