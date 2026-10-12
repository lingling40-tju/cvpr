#!/usr/bin/env bash
set -Eeuo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_sequence_granularity_20261011"
state="$root/runlogs/sequence_evaluation"
result="$root/runlogs/sequence_development256"
mkdir -p "$state" "$result" "$root/runlogs/gpu_eval_locks"
exec 9>"$state/suite.lock"
flock -n 9 || { echo "evaluation suite already active" >&2; exit 2; }
on_exit(){ rc=$?; if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$state/suite.failed"; fi; }
trap on_exit EXIT
rm -f "$state/suite.failed"
cd "$root"
test -f "$root/runlogs/sequence_training/training.completed"
test ! -f "$root/runlogs/sequence_training/sequence.failed"
"$base/activevln_train_env/bin/python" tools/verify_sequence_eval_freeze.py --root "$root"
for method in gspo_sequence gmpo_geomean dhpo_average; do
  test -f "$root/runlogs/seq_${method}_n4_seed11/completed"
  test ! -f "$root/runlogs/seq_${method}_n4_seed11/failed"
  test -f "$root/runlogs/seq_${method}_n4_seed11/train_gradient_audit_64.json"
done
# Stop only this experiment's GPU2 service after checking PID, cwd and port.
if test -s "$root/runlogs/service/server.pid"; then
  pid=$(cat "$root/runlogs/service/server.pid")
  if kill -0 "$pid" 2>/dev/null; then
    cwd=$(readlink "/proc/$pid/cwd")
    args=$(ps -o args= -p "$pid")
    test "$cwd" = "$root"
    printf '%s' "$args" | grep -F 'vlnce_server.server' >/dev/null
    printf '%s' "$args" | grep -F 'server.port=5186' >/dev/null
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then echo "owned Habitat service did not stop" >&2; exit 1; fi
  fi
fi
if curl -fsS --max-time 1 http://127.0.0.1:5186/health >/dev/null 2>&1; then echo "port 5186 remains active" >&2; exit 1; fi
# Two independent model servers fit on GPU0/1; the third follows to avoid shared GPU contention.
bash tools/run_sequence_dev_model.sh gspo_sequence 0 >"$result/seq_gspo_sequence_n4_seed11.launcher.log" 2>&1 &
p0=$!
bash tools/run_sequence_dev_model.sh gmpo_geomean 1 >"$result/seq_gmpo_geomean_n4_seed11.launcher.log" 2>&1 &
p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
bash tools/run_sequence_dev_model.sh dhpo_average 0 >"$result/seq_dhpo_average_n4_seed11.launcher.log" 2>&1
for method in gspo_sequence gmpo_geomean dhpo_average; do
  label="seq_${method}_n4_seed11"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py \
    --root "$result" --manifest prepared_data/development256.json --role development \
    --control positive_initial_sft --candidate "$label" \
    --compact "$state/${method}_vs_sft.jsonl" --output "$state/${method}_vs_sft.json"
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_compact.py \
    --manifest prepared_data/development256.json --compact "$state/${method}_vs_sft.jsonl" \
    --report "$state/${method}_vs_sft.json" --validators "$result" \
    --output "$state/${method}_vs_sft.independent.json"
done
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_sequence_raw_metrics.py \
  --root "$result" --manifest prepared_data/development256.json \
  --methods gspo_sequence gmpo_geomean dhpo_average --output "$state/raw_id_independent_recount.json"
CUDA_VISIBLE_DEVICES="" "$base/activevln_train_env/bin/python" - <<'PY'
import json, math, hashlib
from pathlib import Path
root=Path(".")
state=root/"runlogs/sequence_evaluation"
names=("gspo_sequence","gmpo_geomean","dhpo_average")
pairs={}
gates={}
for name in names:
    report=json.loads((state/(name+"_vs_sft.json")).read_text())
    independent=json.loads((state/(name+"_vs_sft.independent.json")).read_text())
    assert math.isclose(report["paired_sr_points"],independent["paired_sr_points"],abs_tol=1e-10,rel_tol=0)
    assert math.isclose(report["paired_spl_points"],independent["paired_spl_points"],abs_tol=1e-10,rel_tol=0)
    passed=report["paired_sr_points"]>=2.0 and report["paired_spl_points"]>=2.0
    gates[name]={"pass_vs_sft":passed,"eligible_for_scale":passed}
    pairs[name]={"paired_sr_points":report["paired_sr_points"],"paired_spl_points":report["paired_spl_points"],"candidate_successes":report["candidate_successes"],"control_successes":report["control_successes"],"scene_bootstrap_95pct_sr_points":report["scene_bootstrap_95pct_sr_points"],"scene_bootstrap_95pct_spl_points":report["scene_bootstrap_95pct_spl_points"]}
audit=json.loads((state/"raw_id_independent_recount.json").read_text())
assert audit["status"]=="PASS" and audit["raw_ids_checked"]==1024
out={"schema":"sequence_granularity_development_report_v1","episodes":256,"scenes":8,"training_seed":11,"group_size":4,"updates_total_including_smoke":64,"inference_dtype":"float16","same_fp16_sft_reference":True,"manifest_sha256":"8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3","gate_each_metric_points":2.0,"gates":gates,"comparisons":pairs,"raw_audit_sha256":hashlib.sha256((state/"raw_id_independent_recount.json").read_bytes()).hexdigest(),"limits":["One configured training seed.","Development scenes were reused in prior adaptive research and are not a clean test.","The three methods were trained serially due GPU availability; only two evaluation models ran concurrently.","A gate pass would be exploratory and require fresh configured-seed replications."],"reserved_opened":False,"val_unseen_opened":False}
with (state/"sequence_granularity_development_report.json").open("x") as f:f.write(json.dumps(out,indent=2)+"\n")
print(json.dumps({"gates":gates,"comparisons":pairs}))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
