#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
oracle="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
hardneg="$base/policy_stop_hardneg_20261004"
pilot="$oracle/runlogs/oracle_pilot"
run="$oracle/runlogs/oracle_full_recheck"
result="$oracle/runlogs/oracle_full1839"
mkdir -p "$run" "$result"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'oracle full recheck already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
test -f "$pilot/completed" && test -f "$pilot/positive_upperbound"
test ! -f "$pilot/no_pilot_gain"
"$base/activevln_server_env/bin/python" - "$pilot/paired_oracle_vs_control.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['episodes']==256 and x['manifest_sha256']=='bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c'
assert x['candidate_metrics']['successes']==81 and x['control_metrics']['successes']==72
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert x['paired']['sr_pp']>0 and x['paired']['spl_pp']>0
PY
source_manifest="$control/runlogs/three_direction_full_val_unseen/manifest.json"
test "$(sha256sum "$source_manifest" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
cp "$source_manifest" "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
screen_manifest="$oracle/tools/process_val256_manifest.json"
test "$(sha256sum "$screen_manifest" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
candidate=oracle_turnwise_64_seed11_full_recheck
baseline=qwen3_exact_control_64_seed11_oracle_full_recheck
candidate_checkpoint="$oracle/verl_checkpoints/oracle_turnwise_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$control/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json" && test -f "$control_checkpoint/config.json"
for port in 8101 8102; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "full recheck port $port occupied" >&2; exit 1
  fi
done
eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED=11 bash "$control/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}

# Candidate uses GPU 3 for inference and GPU 2 for Habitat while the new
# observation-only reward model fits independently on GPU 1.
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[2]}" -lt 8000 && test "${used[3]}" -lt 8000; then break; fi
  sleep 15
done
eval_lane "$candidate" "$candidate_checkpoint" 3 2 8101 & candidate_pid=$!
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/candidate_started"

# Wait for the GPU-1 representation fit to terminate before the matched
# control evaluation. Its success is not required for the oracle recheck.
while ! test -f "$hardneg/runlogs/full/completed" && \
      ! test -f "$hardneg/runlogs/full/failed"; do
  if test -s "$hardneg/runlogs/after_oracle/launcher.pid" && \
      ! kill -0 "$(cat "$hardneg/runlogs/after_oracle/launcher.pid")" 2>/dev/null; then
    echo 'STOP representation launcher vanished without terminal marker' >&2
    exit 1
  fi
  sleep 30
done
while true; do
  used1=$(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
  test -n "$used1"
  if test "$used1" -lt 8000; then break; fi
  sleep 15
done
eval_lane "$baseline" "$control_checkpoint" 1 0 8102 & control_pid=$!
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/control_started"
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0
test -f "$result/$candidate.completed" && test -f "$result/$baseline.completed"
"$base/activevln_server_env/bin/python" "$control/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 1839 --output "$run/paired_oracle_full.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" "$control/tools/export_matched_episodes.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 1839 --output "$run/paired_oracle_full_episodes.jsonl" \
  >"$run/paired_export.log" 2>&1
PYTHONPATH="$control/tools${PYTHONPATH:+:$PYTHONPATH}" \
  "$base/activevln_server_env/bin/python" "$oracle/tools/analyze_oracle_full_recheck.py" \
  --root "$result" --screen-manifest "$screen_manifest" \
  --candidate "$candidate" --control "$baseline" \
  --full-pair "$run/paired_oracle_full.json" \
  --output "$run/oracle_full_recheck_analysis.json" \
  >"$run/oracle_full_recheck_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_oracle_full.json" "$run" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); run=pathlib.Path(sys.argv[2])
assert x['split']=='val_unseen' and x['episodes']==1839 and x['scenes']==11
assert x['manifest_sha256']=='262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==1839
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
name='positive_full_one_seed' if x['paired']['sr_pp']>0 and x['paired']['spl_pp']>0 else 'no_full_gain'
(run/name).write_text('post-screen privileged one-seed recheck only\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
