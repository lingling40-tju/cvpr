#!/usr/bin/env bash
set -euo pipefail

# CPU only. Freeze replay selection immediately after the staged n=4
# two/three-seed coverage inventory passes; do not start Habitat here.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
pool="$root/runlogs/future_advantage_pooled_preflight"
run="$root/runlogs/future_advantage_sparse_manifest"
dataset="$root/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
split="$base/ActiveVLN_three_directions_20261002/runlogs/ordinal_progress/stop_history_lora_scene_split.json"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'sparse manifest watcher already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$pool/completed"; do
  test ! -f "$pool/failed" || { echo 'pooled coverage watcher failed' >&2; exit 1; }
  test -s "$pool/launcher.pid"
  pid=$(cat "$pool/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_future_advantage_pool_after_scale.sh' >/dev/null || {
      echo 'pooled coverage watcher vanished' >&2; exit 1;
    }
  sleep 60
done
test -s "$pool/report.json"
if ! "$base/activevln_train_env/bin/python" - "$pool/report.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))[
    "enough_coverage_for_fit_preparation"] else 1)
PY
then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_coverage_short"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi

sha256sum "$root/tools/prepare_future_advantage_sparse_manifest.py" \
  "$pool/report.json" "$dataset" "$split" >"$run/source.sha256"
test "$(sha256sum "$root/tools/prepare_future_advantage_sparse_manifest.py" |
  awk '{print $1}')" = \
  372bd7c27e7e1b7b2e30ed7f882d8467eac53ff0e590abb043884fca829872f6
"$base/activevln_train_env/bin/python" \
  "$root/tools/prepare_future_advantage_sparse_manifest.py" \
  --root "$root" --report "$pool/report.json" --dataset "$dataset" \
  --scene-split "$split" --replay-output "$run/replay_manifest.json" \
  --labels-output "$run/pair_labels.json" \
  >"$run/analysis.log" 2>&1
test -s "$run/replay_manifest.json"
test -s "$run/pair_labels.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
