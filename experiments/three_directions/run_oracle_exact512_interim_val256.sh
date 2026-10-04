#!/usr/bin/env bash
set -euo pipefail

# Matched, exploratory step-64 screen while seed-11 step-128 training
# continues on GPUs 2/3. The shared evaluator lock prevents overlap
# with the later complete candidate evaluation on GPU 1.
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turnwise_oracle_20261004"
control="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/oracle_exact512_interim_screen"
result="$root/runlogs/oracle_exact512_interim_val256"
manifest_source="$root/tools/process_val256_manifest.json"
candidate="oracle_turnwise_exact512_seed11_step64_interim"
baseline="oracle_exact512_control_seed11_step64_interim"
candidate_checkpoint="$root/verl_checkpoints/oracle_turnwise_exact512_128_seed11/global_step_64/actor/huggingface"
baseline_checkpoint="$control/verl_checkpoints/oracle_exact512_control_128_seed11/global_step_64/actor/huggingface"
mkdir -p "$run" "$result"
exec 9>"$run/run.lock"
flock -n 9 || { echo 'step-64 screen already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

test -f "$candidate_checkpoint/config.json"
test -f "$baseline_checkpoint/config.json"
test "$(sha256sum "$manifest_source" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
if ! test -f "$result/manifest.json"; then
  cp "$manifest_source" "$result/manifest.json"
fi
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
sha256sum "$root/tools/run_oracle_exact512_interim_val256.sh" \
  "$control/tools/run_direction_eval.sh" \
  "$candidate_checkpoint/config.json" \
  "$baseline_checkpoint/config.json" \
  "$result/manifest.json" >"$run/source.sha256"

for item in "$baseline:$baseline_checkpoint" "$candidate:$candidate_checkpoint"; do
  label=${item%%:*}
  checkpoint=${item#*:}
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT=8140 \
    VLN_VLLM_SEED=11 bash "$control/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" 1 0 \
      >"$run/$label.launcher.log" 2>&1
  test -f "$result/$label.completed"
  test -s "$result/$label.validated.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/$label.completed"
done
"$base/activevln_server_env/bin/python" \
  "$control/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --shards 4 \
  --output "$run/paired_step64.json" >"$run/analysis.log" 2>&1
test -s "$run/paired_step64.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
