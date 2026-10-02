#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_fused_reward_20261003"
run="$root/runlogs/failure_rank"
mkdir -p "$run"
exec 9>"$run/audit.lock"
flock -n 9 || { echo 'failure-rank scene audit already active' >&2; exit 2; }
if test -f "$run/audit.completed"; then exit 0; fi
rm -f "$run/audit.failed"
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then printf '%s\n' "$status" >"$run/audit.failed"; fi
}
trap on_exit EXIT
until test -f "$run/training.completed"; do
  if test -f "$run/training.failed"; then echo 'failure-rank training failed' >&2; exit 1; fi
  pid=$(cat "$run/training_launcher.pid")
  kill -0 "$pid" 2>/dev/null || { echo "training PID $pid stopped" >&2; exit 1; }
  sleep 30
done
decision=$("$base/activevln_train_env/bin/python" - "$run/development.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['schema']=='failure_rank_encoder_development_v1'
assert report['development']['pairs']==125
print('eligible' if all(report['predeclared_gate'].values()) else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/development_decision.txt"
if [ "$decision" = ineligible ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/no_development_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/audit.completed"
  exit 0
fi
test "$decision" = eligible
cd "$source_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$root/tools:$source_root/tools:$source_root/vlnce_server:$source_root${PYTHONPATH:+:$PYTHONPATH}"
nice -n 10 "$base/activevln_train_env/bin/python" \
  "$root/tools/audit_failure_rank_encoder.py" \
  --manifest "$run/manifest.json" \
  --features "$run/features_full/features.pt" \
  --development "$run/development.json" \
  --checkpoint "$run/encoder.pt" \
  --old-encoder "$source_root/runlogs/ordinal_progress/policy_preference/temporal_contrastive_v2/encoder.pt" \
  --old-pair-manifest "$source_root/runlogs/ordinal_progress/policy_preference/manifest.json" \
  --old-v2-manifest "$source_root/runlogs/ordinal_progress/policy_preference/temporal_contrastive_v2/manifest.json" \
  --old-features "$source_root/runlogs/ordinal_progress/policy_preference/navigation_sft_server_prompt/features.pt" \
  --old-swaps "$source_root/runlogs/ordinal_progress/policy_preference/temporal_contrastive_v2/swaps" \
  --output "$run/audit.json" >"$run/audit.log" 2>&1
test -s "$run/audit.json"
decision=$("$base/activevln_train_env/bin/python" - "$run/audit.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['schema']=='failure_rank_encoder_scene_audit_v1'
assert report['failure_rank_candidate']['pairs']==106
assert report['reused_success_endpoint']['pairs']==48
print('eligible' if all(report['predeclared_gate'].values()) else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/audit_decision.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/audit.completed"
