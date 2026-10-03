#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
prior="$base/ActiveVLN_qwen_group_rank_20261004"
root="$base/ActiveVLN_qwen_confident_20261004"
pilot="$prior/runlogs/qwen_group_followup"
scale="$prior/runlogs/qwen_group_scale"
run="$prior/runlogs/qwen_confident_followup"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'confidence follow-up watcher already running' >&2; exit 2; }
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/started"

while ! test -f "$pilot/completed"; do
  if test -f "$pilot/failed"; then
    echo 'original Qwen pilot failed; confidence fallback has no valid gate' >&2
    exit 1
  fi
  if test -s "$pilot/launcher.pid" && \
      ! kill -0 "$(cat "$pilot/launcher.pid")" 2>/dev/null; then
    echo 'original Qwen pilot launcher exited without completion' >&2
    exit 1
  fi
  sleep 60
done
decision=$("$base/activevln_server_env/bin/python" - "$pilot/paired_qwen_vs_control.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256 and x['scenes']==9
assert x['manifest_sha256']=='1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc'
assert x['candidate']=='qwen_group_rank_64_seed11'
assert x['control']=='qwen_exact_control_64_seed11'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
sr,spl=x['paired']['sr_pp'],x['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('positive' if sr>0 and spl>0 else 'negative')
PY
)
printf '%s\n' "$decision" >"$run/prior_pilot_decision.txt"
if test "$decision" = positive; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_positive_prior"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
  exit 0
fi
test "$decision" = negative
while ! test -f "$scale/no_pilot_gain"; do
  if test -f "$scale/suite.failed"; then
    echo 'prior scale gate failed before writing no_pilot_gain' >&2; exit 1
  fi
  if test -s "$scale/launcher.pid" && \
      ! kill -0 "$(cat "$scale/launcher.pid")" 2>/dev/null; then
    echo 'prior scale watcher exited without gate marker' >&2; exit 1
  fi
  sleep 30
done
while ! test -f "$scale/suite.completed"; do
  if test -f "$scale/suite.failed"; then
    echo 'prior scale gate failed after writing no_pilot_gain' >&2; exit 1
  fi
  sleep 2
done
for port in 8031 5031; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "prior reward/Sim service still on port $port" >&2; exit 1
  fi
done

bash "$prior/tools/prepare_confident_tree.sh" >"$run/tree_prepare.log" 2>&1
test -f "$root/runlogs/tree_provenance.txt"
bash "$root/tools/start_confident_services.sh" >"$run/services.log" 2>&1

for steps in 2 64; do
  bash "$root/tools/run_confident_train.sh" "$steps" \
    >"$run/train_${steps}step.launcher.log" 2>&1
  "$base/activevln_train_env/bin/python" \
    "$root/tools/audit_confident_train.py" \
    --root "$root" --source-root "$source_root" --steps "$steps" \
    --output "$run/train_audit_${steps}step.json" \
    >"$run/train_audit_${steps}step.log" 2>&1
  test -s "$run/train_audit_${steps}step.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/train_${steps}step_audited"
done

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
stop_service confident_habitat 5033 server.port=5033 \
  "$root/runlogs/confident_services/habitat.pid"
stop_service qwen_teacher 8033 qwen_group_rank_server.py \
  "$root/runlogs/confident_services/reward.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/services_stopped"

result="$root/runlogs/confident_val256"
mkdir -p "$result"
cp "$prior/tools/next_val256_manifest.json" "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1
candidate_checkpoint="$root/verl_checkpoints/qwen_confident_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$source_root/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"

eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=256 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED=11 \
    bash "$source_root/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
eval_lane qwen_confident_64_seed11 "$candidate_checkpoint" 3 2 8061 &
candidate_pid=$!
eval_lane qwen_exact_control_64_seed11 "$control_checkpoint" 1 0 8062 &
control_pid=$!
status=0
wait "$candidate_pid" || status=1
wait "$control_pid" || status=1
test "$status" -eq 0
test -f "$result/qwen_confident_64_seed11.completed"
test -f "$result/qwen_exact_control_64_seed11.completed"
"$base/activevln_server_env/bin/python" \
  "$source_root/tools/analyze_matched_pair.py" \
  --root "$result" --candidate qwen_confident_64_seed11 \
  --control qwen_exact_control_64_seed11 --expected-count 256 \
  --output "$run/paired_confident_vs_control.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_confident_vs_control.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256 and x['scenes']==8
assert x['manifest_sha256']=='e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1'
assert x['candidate']=='qwen_confident_64_seed11'
assert x['control']=='qwen_exact_control_64_seed11'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert math.isfinite(x['paired']['sr_pp']) and math.isfinite(x['paired']['spl_pp'])
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
