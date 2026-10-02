#!/usr/bin/env bash
set -euo pipefail

# Small group-four data screen on GPU 0. The audit and initial-state passes are
# skipped unless earlier frozen gates pass. No policy training is launched.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/policy_preference/full_history_probe"
source_dir="$root/runlogs/ordinal_progress/policy_preference"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'full-history probe already running' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() {
  rc=$?
  if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$run/suite.failed"; fi
}
trap on_exit EXIT
cd "$root"
test "$(sha256sum "$source_dir/manifest.json" | awk '{print $1}')" = \
  cc3cb63c0ae1ae3c9b47a9d191dbf548d9dce12e08feed0ec8b314a1b7398c47
test "$(sha256sum "$source_dir/full_history_probe_manifest.json" | awk '{print $1}')" = \
  2905c5d2da6631ea9ac0ef64eb1929b6b308d8c9ed37b392a79b3477bdaafe3e
test "$(sha256sum "$model/config.json" | awk '{print $1}')" = \
  9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9
test -f "$source_dir/navigation_sft_server_prompt.completed"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
server_python="$base/activevln_server_env/bin/python"
train_python="$base/activevln_train_env/bin/python"
probe="$source_dir/full_history_probe_manifest.json"
records="$run/frames"
scores="$run/scores"
for split in development audit; do
  if test "$split" = audit; then
    "$train_python" - "$run/development_analysis.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['predeclared_screen_gate']['passes'], 'development gate failed'
PY
  fi
  "$server_python" tools/collect_full_history_probe.py \
    --manifest "$source_dir/manifest.json" --probe-manifest "$probe" \
    --output "$records" --split "$split" --gpu 0 \
    >"$run/collect_${split}.log" 2>&1
  "$train_python" tools/score_full_history_probe.py \
    --probe-manifest "$probe" --records-root "$records" --model "$model" \
    --output-root "$scores" --split "$split" --phase terminal \
    >"$run/score_${split}_terminal.log" 2>&1
  "$train_python" tools/analyze_full_history_probe.py \
    --probe-manifest "$probe" --source-manifest "$source_dir/manifest.json" \
    --scores-root "$scores" \
    --single-observation-features "$source_dir/navigation_sft_server_prompt/features.pt" \
    --split "$split" --output "$run/${split}_analysis.json" \
    >"$run/analyze_${split}.log" 2>&1
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/${split}_terminal.completed"
  if test "$split" = development; then
    if ! "$train_python" - "$run/development_analysis.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data['predeclared_screen_gate']['passes']
PY
    then
      printf 'development ranking below 70%%; audit and RL skipped\n' >"$run/decision.txt"
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
      exit 0
    fi
  fi
done
"$train_python" - "$run/audit_analysis.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
gate=data['predeclared_screen_gate']
assert 'accuracy_at_least_0_75' in gate
PY
if "$train_python" - "$run/audit_analysis.json" <<'PY'
import json,sys
gate=json.load(open(sys.argv[1]))['predeclared_screen_gate']
assert gate['accuracy_at_least_0_75'] and gate['reference_gain_at_least_5pp']
PY
then
  "$train_python" tools/score_full_history_probe.py \
    --probe-manifest "$probe" --records-root "$records" --model "$model" \
    --output-root "$scores" --split audit --phase initial \
    >"$run/score_audit_initial.log" 2>&1
  "$train_python" tools/analyze_full_history_probe.py \
    --probe-manifest "$probe" --source-manifest "$source_dir/manifest.json" \
    --scores-root "$scores" \
    --single-observation-features "$source_dir/navigation_sft_server_prompt/features.pt" \
    --split audit --output "$run/audit_analysis.json" \
    >"$run/analyze_audit_progress.log" 2>&1
  printf 'terminal ranking passed; inspect progress and matched-prompt ablation before RL\n' >"$run/decision.txt"
else
  printf 'audit terminal ranking failed; initial-state inference and RL skipped\n' >"$run/decision.txt"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
