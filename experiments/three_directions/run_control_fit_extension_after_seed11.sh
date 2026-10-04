#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
oracle="$base/ActiveVLN_turnwise_oracle_20261004"
scratch="$base/policy_control_fit_extension_20261004"
run="$scratch/runlogs/fit_extension"
ids="$scratch/control_exact512_fit_extension_ids.json"
old="$base/policy_action_memory_lora_20261004/diversity_fit1024_manifest.json"
source="$root/verl_checkpoints/oracle_exact512_control_128_seed11/rollout.jsonl"
source_done="$root/runlogs/oracle_exact512_control_128_seed11/completed"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'control fit extension already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
test "$(sha256sum "$ids" | awk '{print $1}')" = \
  d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081
test "$(sha256sum "$old" | awk '{print $1}')" = \
  bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37
suite="$oracle/runlogs/oracle_exact512_scale"
while ! test -f "$source_done"; do
  if test -f "$root/runlogs/oracle_exact512_control_128_seed11/failed" || \
     test -f "$suite/suite.failed"; then
    echo 'source control training failed' >&2; exit 1
  fi
  test -s "$suite/launcher.pid"
  pid=$(cat "$suite/launcher.pid")
  if ! ps -o args= -p "$pid" | grep -F \
      'run_oracle_exact512_scale_after_full.sh' >/dev/null; then
    echo 'source scale suite vanished before seed11 completed' >&2; exit 1
  fi
  sleep 30
done
test -s "$source"
while true; do
  used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    sed -n '2p' | tr -d ' ')
  test -n "$used1"
  if test "$used1" -lt 8000; then break; fi
  sleep 30
done
cd "$root"
export PYTHONPATH="$scratch:$root/tools:$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_server_env/bin/python"
sha256sum "$scratch"/*.py "$scratch"/*.sh >"$run/source.sha256"
"$python" "$scratch/prepare_control_fit_all_variants.py" \
  --ids "$ids" --old-manifest "$old" \
  --train-dataset data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --control-rollout "$source" \
  --output "$scratch/all_variant_manifest.json" \
  >"$run/prepare.log" 2>&1
"$python" "$scratch/collect_control_fit_labels.py" \
  --manifest "$scratch/all_variant_manifest.json" \
  --output "$scratch/label_smoke" --gpu 1 --limit 4 \
  >"$run/label_smoke.log" 2>&1
"$python" - "$scratch/label_smoke/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['requested']==x['completed']==4 and x['rgb_frames_written']==0
PY
"$python" "$scratch/collect_control_fit_labels.py" \
  --manifest "$scratch/all_variant_manifest.json" \
  --output "$scratch/label_only" --gpu 1 \
  >"$run/label_only.log" 2>&1
"$python" "$scratch/select_control_fit_render.py" \
  --ids "$ids" --all-manifest "$scratch/all_variant_manifest.json" \
  --labels-root "$scratch/label_only" \
  --report "$run/selection.json" \
  --render-manifest "$scratch/render_manifest.json" \
  >"$run/selection.log" 2>&1
if ! "$python" - "$run/selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['schema']=='control_fit_render_selection_v1'
sys.exit(0 if x['sample_gate']['passed'] else 1)
PY
then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/failed_sample_gate"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
rgb_start=$(date +%s)
"$python" tools/collect_policy_process_turns.py \
  --manifest "$scratch/render_manifest.json" --part fit \
  --output-root "$scratch/render_turns" --gpu 1 \
  >"$run/rgb_replay.log" 2>&1
rgb_end=$(date +%s)
echo "$((rgb_end-rgb_start))" >"$run/rgb_replay_seconds.txt"
"$python" "$scratch/audit_control_fit_render.py" \
  --ids "$ids" --label-manifest "$scratch/all_variant_manifest.json" \
  --labels-root "$scratch/label_only" \
  --selection-report "$run/selection.json" \
  --render-manifest "$scratch/render_manifest.json" \
  --render-root "$scratch/render_turns" \
  --rgb-seconds-file "$run/rgb_replay_seconds.txt" \
  --output "$run/fit_audit.json" \
  >"$run/audit.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/passed_sample_gate"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
