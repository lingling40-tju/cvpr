#!/usr/bin/env bash
# Resume the failed evaluation from completed, audited pilot checkpoints.
# This entry point never invokes smoke tests or training.
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/positive_pilot"
result="$root/runlogs/positive_development256"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'pilot suite still active' >&2; exit 2; }
test -f "$state/training.completed"
test ! -f "$root/runlogs/positive_scale/reserved.opened"
if test -f "$state/suite.completed" && ! test -f "$state/suite.failed"; then exit 0; fi
for arm in control candidate; do
  label="positive_trajectory_${arm}_64step_seed11"
  test -f "$root/runlogs/$label/completed"
  test -f "$root/verl_checkpoints/$label/global_step_64/actor/huggingface/config.json"
  "$base/activevln_train_env/bin/python" tools/audit_positive_train.py \
    --log "$root/runlogs/$label/train.log" --steps 64 --arm "$arm" \
    --output "$state/${arm}_train_audit.json" >"$state/${arm}_train_audit.log"
done
"$base/activevln_server_env/bin/python" tools/eval_train_scene_subset.py \
  --model-label configuration_preflight --manifest "$root/prepared_data/development256.json" \
  --result-root "$result" --role development --validate-only \
  >"$state/repaired_ndtw_preflight.json"

# Preserve all partial episodes and failed logs; restart both arms on the
# unchanged manifest with the same checkpoint and generation parameters.
"$base/activevln_server_env/bin/python" - "$root" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import re
import sys
root = Path(sys.argv[1])
state = root / "runlogs/positive_pilot"
result = root / "runlogs/positive_development256"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archive = state / ("failed_ndtw_reference_" + stamp)
archive.mkdir()
stats = list(result.glob("**/stats_*.json")) if result.exists() else []
logs = list(result.glob("eval_*.log")) if result.exists() else []
record = {
    "schema": "positive_pilot_ndtw_reference_recovery_v1",
    "utc": stamp,
    "cause": "DATASET.SPLIT=train but NDTW.SPLIT remained val_unseen",
    "repair": "NDTW.SPLIT=train; require all frozen episode reference locations",
    "partial_stat_files_archived": len(stats),
    "partial_unique_ids_archived": sorted({str(json.loads(p.read_text())["id"]) for p in stats}),
    "failed_shard_logs": {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                 "key_errors": re.findall(r"KeyError: '([^']+)'", p.read_text(errors="replace"))}
        for p in logs
    },
    "repaired_evaluator_sha256": hashlib.sha256((root / "tools/eval_train_scene_subset.py").read_bytes()).hexdigest(),
    "reference_preflight": json.loads((state / "repaired_ndtw_preflight.json").read_text()),
    "training_restarted": False,
    "manifest_checkpoint_policy_metric_formula_gate_unchanged": True,
    "archive": str(archive),
}
if result.exists(): os.replace(result, archive / "positive_development256")
for name in ("suite.failed", "development_pair.failed", "development_pair.completed",
             "control_eval.launcher.log", "candidate_eval.launcher.log",
             "control_parallel_eval.launcher.log", "candidate_parallel_eval.launcher.log"):
    p = state / name
    if p.exists(): os.replace(p, archive / name)
result.mkdir()
(state / "ndtw_reference_recovery.json").write_text(json.dumps(record, indent=2) + "\n")
PY
cleanup() {
  status=$?
  if test "$status" -ne 0; then
    rm -f "$state/suite.completed"
    printf '%s\n' "$status" >"$state/suite.failed"
  fi
}
trap cleanup EXIT
bash tools/run_positive_development_eval.sh control >"$state/recovered_eval.launcher.log" 2>&1
"$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py \
  --root "$result" --manifest "$root/prepared_data/development256.json" \
  --role development --control positive_trajectory_control_64step_seed11 \
  --candidate positive_trajectory_candidate_64step_seed11 \
  --compact "$state/development256_paired.jsonl" \
  --output "$state/development256_pair.json" >"$state/analyze_pair.log"
"$base/activevln_server_env/bin/python" - "$state/development256_pair.json" "$state/frozen_gate.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
gate = {"schema": "positive_trajectory_frozen_development_gate_v1",
        "paired_sr_points": report["paired_sr_points"],
        "paired_spl_points": report["paired_spl_points"],
        "requires_each_metric_points_at_least": 2.0,
        "pass": report["paired_sr_points"] >= 2.0 and report["paired_spl_points"] >= 2.0,
        "reserved_screen_opened": False}
with open(sys.argv[2], "w") as handle: json.dump(gate, handle, indent=2)
print(json.dumps(gate))
PY
"$base/activevln_server_env/bin/python" tools/verify_positive_compact.py \
  --manifest "$root/prepared_data/development256.json" \
  --compact "$state/development256_paired.jsonl" --report "$state/development256_pair.json" \
  --validators "$result" --gate "$state/frozen_gate.json" \
  --output "$state/independent_compact_recount.json" >"$state/independent_recount.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
