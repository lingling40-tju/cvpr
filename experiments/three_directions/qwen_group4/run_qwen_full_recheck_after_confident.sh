#!/usr/bin/env bash
set -euo pipefail

# Exploratory sensitivity check: a prior n=4 checkpoint lost on 256 cases
# but improved on 1,839. This reuses the already trained Qwen checkpoints.
# It starts only if the subsequent confidence pilot also misses its gate.
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
prior="$base/ActiveVLN_qwen_group_rank_20261004"
confidence="$base/ActiveVLN_qwen_confident_20261004"
original="$prior/runlogs/qwen_group_followup"
pilot="$prior/runlogs/qwen_confident_followup"
scale="$prior/runlogs/qwen_confident_scale_watcher"
run="$prior/runlogs/qwen_full_recheck"
mkdir -p "$run"
exec 9>"$run/suite.lock"
flock -n 9 || { echo 'Qwen full recheck already active' >&2; exit 2; }
if test -f "$run/suite.completed"; then exit 0; fi
rm -f "$run/suite.failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/suite.failed"; fi; }
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.started"

test -f "$original/completed"
"$base/activevln_server_env/bin/python" - "$original/paired_qwen_vs_control.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256
assert x['manifest_sha256']=='1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc'
assert x['candidate']=='qwen_group_rank_64_seed11' and x['control']=='qwen_exact_control_64_seed11'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert x['paired']['sr_pp']<0 and x['paired']['spl_pp']<0
for metric in ('sr_pp_95','spl_pp_95'):
    low,high=x['paired']['scene_cluster_bootstrap95'][metric]
    assert low<0<high
PY

while ! test -f "$pilot/completed"; do
  if test -f "$pilot/failed"; then
    echo 'confidence pilot failed; repair it before full recheck' >&2; exit 1
  fi
  if test -s "$pilot/launcher.pid" && \
      ! kill -0 "$(cat "$pilot/launcher.pid")" 2>/dev/null; then
    echo 'confidence launcher exited without completion' >&2; exit 1
  fi
  sleep 60
done
decision=$("$base/activevln_server_env/bin/python" - "$pilot/paired_confident_vs_control.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==256 and x['scenes']==8
assert x['manifest_sha256']=='e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1'
assert x['candidate']=='qwen_confident_64_seed11' and x['control']=='qwen_exact_control_64_seed11'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==256
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
sr,spl=x['paired']['sr_pp'],x['paired']['spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
print('skip_confidence_positive' if sr>0 and spl>0 else 'recheck')
PY
)
printf '%s\n' "$decision" >"$run/confidence_decision.txt"
if test "$decision" = skip_confidence_positive; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/skipped_confidence_positive"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
  exit 0
fi
test "$decision" = recheck
while ! test -f "$scale/no_pilot_gain" || ! test -f "$scale/suite.completed"; do
  if test -f "$scale/suite.failed"; then
    echo 'confidence scale gate failed' >&2; exit 1
  fi
  if test -s "$scale/launcher.pid" && \
      ! kill -0 "$(cat "$scale/launcher.pid")" 2>/dev/null; then
    echo 'confidence scale watcher exited without negative gate' >&2; exit 1
  fi
  sleep 30
done
for port in 8033 5033 8061 8062 8081 8082; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || \
      curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "port $port still in use" >&2; exit 1
  fi
done
test -f "$confidence/runlogs/confident_val256/qwen_confident_64_seed11.completed"

result="$prior/runlogs/qwen_full_recheck1839"
mkdir -p "$result"
cp "$source_root/runlogs/three_direction_full_val_unseen/manifest.json" \
  "$result/manifest.json"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e
candidate=qwen_group_rank_64_seed11_full_recheck
control=qwen_exact_control_64_seed11_full_recheck
candidate_checkpoint="$prior/verl_checkpoints/qwen_group_rank_64step_seed11/global_step_64/actor/huggingface"
control_checkpoint="$source_root/verl_checkpoints/qwen3_exact_control_64step_seed11/global_step_64/actor/huggingface"
test -f "$candidate_checkpoint/config.json"
test -f "$control_checkpoint/config.json"

eval_lane() {
  local label=$1 checkpoint=$2 model_gpu=$3 sim_gpu=$4 port=$5
  VLN_EVAL_RESULT_ROOT="$result" VLN_EVAL_MANIFEST="$result/manifest.json" \
    VLN_EVAL_COUNT=1839 VLN_EVAL_SHARDS=4 VLN_EVAL_PORT="$port" \
    VLN_VLLM_SEED=11 \
    bash "$source_root/tools/run_direction_eval.sh" \
      "$label" "$checkpoint" "$model_gpu" "$sim_gpu" \
      >"$run/$label.eval_launcher.log" 2>&1
}
eval_lane "$candidate" "$candidate_checkpoint" 3 2 8081 & candidate_pid=$!
eval_lane "$control" "$control_checkpoint" 1 0 8082 & control_pid=$!
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
  --output "$run/paired_qwen_full_recheck.json" \
  >"$run/paired_analysis.log" 2>&1
"$base/activevln_server_env/bin/python" - "$run/paired_qwen_full_recheck.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['split']=='val_unseen' and x['episodes']==1839 and x['scenes']==11
assert x['manifest_sha256']=='262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e'
assert x['candidate_metrics']['count']==x['control_metrics']['count']==1839
assert x['candidate_metrics']['inference_errors']==x['control_metrics']['inference_errors']==0
assert math.isfinite(x['paired']['sr_pp']) and math.isfinite(x['paired']['spl_pp'])
PY
PYTHONPATH="$source_root/tools${PYTHONPATH:+:$PYTHONPATH}" \
  "$base/activevln_server_env/bin/python" \
  "$prior/tools/analyze_qwen_full_recheck.py" \
  --root "$result" \
  --screen-manifest "$prior/runlogs/qwen_group_val256/manifest.json" \
  --candidate "$candidate" --control "$control" \
  --full-pair "$run/paired_qwen_full_recheck.json" \
  --output "$run/qwen_full_recheck_analysis.json" \
  >"$run/qwen_full_recheck_analysis.log" 2>&1
test -s "$run/qwen_full_recheck_analysis.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/suite.completed"
