#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
cache="$root/runlogs/ordinal_progress/group_relative_cache"
run="$root/runlogs/ordinal_progress/group_relative_head"
checkpoint="$root/runlogs/ordinal_progress/history_grounding_lora_seed11/interim_selected.pt"
mkdir -p "$run"
exec 9>"$run/head.lock"
flock -n 9 || { echo 'group-relative head already running' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
for _ in $(seq 1 120); do
  if test -f "$cache/failed"; then
    echo 'feature cache failed' >&2
    exit 1
  fi
  if test -f "$cache/completed"; then break; fi
  sleep 30
done
test -f "$cache/completed"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
"$base/activevln_train_env/bin/python" tools/fit_group_relative_head.py \
  --manifest runlogs/ordinal_progress/group_relative_manifest.json \
  --turn-root runlogs/ordinal_progress/group_relative_turns \
  --cache-root runlogs/ordinal_progress/group_relative_states \
  --cache-audit "$cache/cache_audit.json" \
  --source-id "$(sha256sum "$checkpoint" | awk '{print $1}')" \
  --output "$run" >"$run/train.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
