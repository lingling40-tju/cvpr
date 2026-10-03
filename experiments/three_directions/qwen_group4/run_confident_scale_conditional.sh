#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
prior="$base/ActiveVLN_qwen_group_rank_20261004"
root="$base/ActiveVLN_qwen_confident_20261004"
pilot="$prior/runlogs/qwen_confident_followup"
run="$prior/runlogs/qwen_confident_scale_watcher"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'confidence scale watcher already active' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/suite.failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.started"

while ! test -f "$pilot/completed"; do
  if test -f "$pilot/failed"; then
    echo 'confidence pilot failed; scale cannot start' >&2; exit 1
  fi
  if test -s "$pilot/launcher.pid" && \
      ! kill -0 "$(cat "$pilot/launcher.pid")" 2>/dev/null; then
    echo 'confidence pilot launcher exited without completion' >&2; exit 1
  fi
  sleep 60
done
if test -f "$pilot/skipped_positive_prior"; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_positive_prior"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test -d "$root"
decision=$("$base/activevln_server_env/bin/python" - "$pilot/paired_confident_vs_control.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256 and x['scenes']==8
assert x['manifest_sha256']=='e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1'
assert x['candidate']=='qwen_confident_64_seed11'
assert x['control']=='qwen_exact_control_64_seed11'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
sr,spl=x['paired']['sr_pp'],x['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('eligible' if sr>0 and spl>0 else 'ineligible')
PY
)
printf '%s\n' "$decision" >"$run/pilot_decision.txt"
if test "$decision" = ineligible; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/no_pilot_gain"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test "$decision" = eligible
for tree in "$source_root" "$root"; do
  test "$(sha256sum "$tree/data/qwen3_group4_exact512.parquet" | awk '{print $1}')" = \
    d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f
done
scale_run="$root/runlogs/confident_scale"
mkdir -p "$scale_run"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$scale_run/pilot_eligible"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/pilot_eligible"

stop_service() {
  local name=$1 port=$2 expected=$3 pid_file=$4 pid
  test -s "$pid_file"
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    ps -o args= -p "$pid" | grep -F "$expected" >/dev/null || {
      echo "refusing to stop unmatched $name PID $pid" >&2; return 1;
    }
    kill "$pid"
  fi
  for attempt in $(seq 1 60); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "$name port $port did not stop" >&2
  return 1
}

bash "$source_root/tools/start_group4_service.sh" 0 >"$run/control_service.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_confident_scale_training.sh" control "$seed" \
    >"$run/control_seed${seed}.launcher.log" 2>&1
  test -f "$source_root/runlogs/qwen_confident_scale_control_128step_seed${seed}/completed"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/control_seed${seed}.completed"
done
stop_service control_habitat 5013 server.port=5013 \
  "$source_root/runlogs/group4_service/server.pid"

bash "$root/tools/start_confident_scale_services.sh" \
  >"$run/candidate_services.log" 2>&1
for seed in 11 22 33; do
  bash "$root/tools/run_confident_scale_training.sh" candidate "$seed" \
    >"$run/candidate_seed${seed}.launcher.log" 2>&1
  test -f "$root/runlogs/qwen_confident_scale_128step_seed${seed}/completed"
  "$base/activevln_train_env/bin/python" \
    "$root/tools/audit_confident_scale_training.py" \
    --source-root "$source_root" --candidate-root "$root" --seed "$seed" \
    --output "$run/train_audit_seed${seed}.json" \
    >"$run/train_audit_seed${seed}.log" 2>&1
  test -s "$run/train_audit_seed${seed}.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_seed${seed}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/training.completed"
stop_service candidate_habitat 5034 server.port=5034 \
  "$root/runlogs/confident_scale_services/habitat.pid"
stop_service qwen_teacher 8034 qwen_group_rank_server.py \
  "$root/runlogs/confident_scale_services/reward.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services_stopped"

result="$root/runlogs/confident_full1839"
mkdir -p "$result"
cp "$source_root/runlogs/three_direction_full_val_unseen/manifest.json" \
  "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5 seed=$6
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED="$seed" \
    bash "$source_root/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
for seed in 11 22 33; do
  candidate="qwen_confident_scale_128_seed${seed}"
  control="qwen_confident_scale_control_128_seed${seed}"
  candidate_checkpoint="$root/verl_checkpoints/qwen_confident_scale_128step_seed${seed}/global_step_128/actor/huggingface"
  control_checkpoint="$source_root/verl_checkpoints/qwen_confident_scale_control_128step_seed${seed}/global_step_128/actor/huggingface"
  test -f "$candidate_checkpoint/config.json"
  test -f "$control_checkpoint/config.json"
  eval_lane "$candidate" "$candidate_checkpoint" 3 2 8071 "$seed" & candidate_pid=$!
  eval_lane "$control" "$control_checkpoint" 1 0 8072 "$seed" & control_pid=$!
  status=0
  wait "$candidate_pid" || status=1
  wait "$control_pid" || status=1
  test "$status" -eq 0
  test -f "$result/$candidate.completed"
  test -f "$result/$control.completed"
  "$base/activevln_server_env/bin/python" \
    "$source_root/tools/analyze_matched_pair.py" \
    --root "$result" --candidate "$candidate" --control "$control" \
    --expected-count 1839 \
    --output "$result/paired_${candidate}_vs_${control}.json" \
    >"$run/paired_seed${seed}.log" 2>&1
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/eval_seed${seed}.completed"
done
PYTHONPATH="$source_root/tools:$root/tools${PYTHONPATH:+:$PYTHONPATH}" \
  "$base/activevln_server_env/bin/python" \
  "$root/tools/analyze_confident_full_scale.py" \
  --root "$result" \
  --screen-manifest "$root/tools/next_val256_manifest.json" \
  --output "$run/full1839_analysis.json" \
  >"$run/full1839_analysis.log" 2>&1
test -s "$run/full1839_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$scale_run/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
