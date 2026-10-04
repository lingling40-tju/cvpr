#!/usr/bin/env bash
set -euo pipefail

# Wait for the ongoing full evaluation, then run one n=4 mechanism pilot.
base=/Knowin/foundation/haozhiwang/whz
prior="$base/ActiveVLN_turnwise_oracle_20261004"
root="$base/ActiveVLN_stop_boundary_20261005"
control="$base/ActiveVLN_three_directions_20261002"
scale="$prior/runlogs/oracle_exact512_scale"
run="$root/runlogs/stop_boundary_pilot"
result="$root/runlogs/stop_boundary_val256"
mkdir -p "$run" "$result"
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'stop-boundary pilot already active' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
service_started=0
stop_service() {
  local pid_file="$root/runlogs/stop_boundary_service/habitat.pid" pid
  if test "$service_started" -ne 1 || ! test -s "$pid_file"; then return 0; fi
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F 'server.port=5036' >/dev/null || {
      echo "refusing to stop unmatched PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5036/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo 'stop-boundary Habitat port 5036 did not stop' >&2
  return 1
}
on_exit() {
  local status=$?
  stop_service || status=1
  if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$scale/suite.completed"; do
  test ! -f "$scale/suite.failed" || {
    echo 'three-seed full evaluation failed; diagnose before boundary pilot' >&2
    exit 1
  }
  test -s "$scale/launcher.pid"
  pid=$(cat "$scale/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_exact512_scale_after_full.sh' >/dev/null || {
      echo 'three-seed full evaluation launcher vanished' >&2; exit 1;
    }
  sleep 30
done
test -s "$scale/full1839_analysis.json"
early_pair="$prior/runlogs/oracle_exact512_early_pair_analysis"
while ! test -f "$early_pair/completed"; do
  test ! -f "$early_pair/failed" || {
    echo 'early paired audit failed; diagnose before boundary pilot' >&2
    exit 1
  }
  test -s "$early_pair/launcher.pid"
  pid=$(cat "$early_pair/launcher.pid")
  ps -o args= -p "$pid" | grep -F \
    'run_oracle_exact512_early_pair_analysis.sh' >/dev/null || {
      echo 'early paired audit launcher vanished' >&2; exit 1;
    }
  sleep 30
done
test -f "$control/runlogs/qwen3_exact_control_64step_seed11/completed"
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[1]}" -lt 8000 && \
      test "${used[2]}" -lt 8000 && test "${used[3]}" -lt 8000; then
    break
  fi
  sleep 20
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/gpus_released"

service_started=1
bash "$root/tools/start_stop_boundary_service.sh" \
  >"$run/service_launcher.log" 2>&1
for steps in 2 64; do
  bash "$root/tools/run_stop_boundary_train.sh" "$steps" \
    >"$run/train_${steps}step.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" \
    "$root/tools/audit_stop_boundary_train.py" \
    --root "$root" --control-root "$control" --steps "$steps" \
    --output "$run/train_audit_${steps}step.json" \
    >"$run/train_audit_${steps}step.log" 2>&1
  test -s "$run/train_audit_${steps}step.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_${steps}step_audited"
done
stop_service
service_started=0
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  test "${#used[@]}" -eq 4
  if test "${used[0]}" -lt 8000 && test "${used[1]}" -lt 8000 && \
      test "${used[2]}" -lt 8000 && test "${used[3]}" -lt 8000; then
    break
  fi
  sleep 20
done

manifest="$prior/tools/process_val256_manifest.json"
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
cp "$manifest" "$result/manifest.json"
candidate=stop_boundary_64_seed11
baseline=qwen3_exact_control_64_seed11_oracle_screen
candidate_checkpoint="$root/verl_checkpoints/stop_boundary_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$control/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"
prior_screen="$prior/runlogs/oracle_val256"
test "$(sha256sum "$prior_screen/manifest.json" | awk '{print $1}')" = \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c
test -f "$prior_screen/$baseline.completed"
test "$(sha256sum "$control_checkpoint/config.json" | awk '{print $1}')" = \
  1b069d1d32f459f957e9516f790429b9ef0cd6c9b36e0844768e333c08b9ab67
mkdir -p "$result/$baseline"
cp -a "$prior_screen/$baseline/." "$result/$baseline/"
cp "$prior_screen/$baseline.completed" "$result/$baseline.completed"
"$base/activevln_server_env/bin/python" \
  "$control/tools/validate_full_label.py" \
  "$baseline" "$result" "$result/manifest.json" 4 \
  >"$run/reused_control_validation.json"
printf 'source=%s\ncheckpoint_config_sha256=%s\nmanifest_sha256=%s\n' \
  "$prior_screen/$baseline" \
  1b069d1d32f459f957e9516f790429b9ef0cd6c9b36e0844768e333c08b9ab67 \
  bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c \
  >"$run/reused_control_provenance.txt"
VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
  VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT=8131 \
  VLN_VLLM_SEED=11 bash "$control/tools/run_direction_eval.sh" \
  "$candidate" "$candidate_checkpoint" 3 2 \
  >"$run/$candidate.eval_launcher.log" 2>&1
test -f "$result/$candidate.completed"
test -f "$result/$baseline.completed"
"$base/activevln_server_env/bin/python" \
  "$control/tools/analyze_matched_pair.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_boundary_vs_control.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" \
  "$control/tools/export_matched_episodes.py" \
  --root "$result" --candidate "$candidate" --control "$baseline" \
  --expected-count 256 --output "$run/paired_boundary_episodes.jsonl" \
  >"$run/paired_export.log" 2>&1
"$base/activevln_server_env/bin/python" - \
  "$run/paired_boundary_vs_control.json" "$run" <<'PY'
import json, math, pathlib, sys
x = json.load(open(sys.argv[1]))
run = pathlib.Path(sys.argv[2])
assert x['split'] == 'val_unseen' and x['episodes'] == 256
assert x['manifest_sha256'] == 'bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c'
assert x['candidate_metrics']['count'] == x['control_metrics']['count'] == 256
assert x['candidate_metrics']['inference_errors'] == x['control_metrics']['inference_errors'] == 0
assert all(math.isfinite(x['paired'][metric]) for metric in ('sr_pp', 'spl_pp'))
decision = 'positive_exploratory_screen' if all(x['paired'][metric] > 0
    for metric in ('sr_pp', 'spl_pp')) else 'no_pilot_gain'
(run / decision).write_text('reused val-unseen development screen\n')
print(decision, x['paired']['sr_pp'], x['paired']['spl_pp'])
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
