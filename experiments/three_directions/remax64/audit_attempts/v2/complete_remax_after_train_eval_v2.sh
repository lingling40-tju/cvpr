#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/Knowin/foundation/haozhiwang/whz
ROOT=$BASE/ActiveVLN_remax_20261010
TRAIN=$ROOT/runlogs/remax_n4_seed11
STATE=$ROOT/runlogs/remax_reserved_eval_v2
RESULT=$ROOT/runlogs/reserved256
SFTROOT=$BASE/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_reserved
CHECKPOINT=$ROOT/verl_checkpoints/remax_n4_seed11/global_step_64/actor/huggingface
LABEL=remax_n4_seed11
PORT=8153
mkdir -p "$STATE" "$RESULT" "$ROOT/runlogs/gpu_eval_locks"
exec 9>"$STATE/suite.lock"
flock -n 9 || { echo "ReMax post-train suite already active" >&2; exit 2; }
if test -f "$STATE/suite.completed"; then exit 0; fi
test ! -f "$STATE/suite.failed"
server_pid=""
worker_pids=()
cleanup() {
  rc=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  if test -n "$server_pid"; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  if test "$rc" -ne 0 && ! test -f "$STATE/suite.completed"; then printf '%s\n' "$rc" >"$STATE/suite.failed"; fi
}
trap cleanup EXIT
cd "$ROOT"
"$BASE/activevln_train_env/bin/python" "$STATE/verify_remax_eval_freeze.py" "$ROOT" "$STATE/evaluation_identity.json" "$STATE/evaluation_identity.sha256" "$STATE/complete_remax_after_train_eval_v2.sh" >"$STATE/evaluation_freeze_before.json"
test -s "$TRAIN/train64_launcher.pid"
train_pid=$(cat "$TRAIN/train64_launcher.pid")
while ! test -f "$TRAIN/completed"; do
  test ! -f "$TRAIN/failed"
  if ! kill -0 "$train_pid" 2>/dev/null; then
    sleep 5
    test -f "$TRAIN/completed" || { echo "training owner exited without completed marker" >&2; exit 1; }
  fi
  sleep 30
done
flock "$TRAIN/run.lock" -c true
test ! -f "$TRAIN/failed"
test "$(sha256sum "$ROOT/runlogs/freeze/identity.json" | awk '{print $1}')" = d7a218037ea5e8ca7ad775df64ebcd9fdb22dcfd48af1f32e340e244faec207f
(cd "$ROOT"; sha256sum -c runlogs/freeze/source_files.sha256 >/dev/null)
expected_protocol=$(python -c 'import json; print(json.load(open("runlogs/freeze/identity.json"))["protocol_sha256"])')
actual_protocol=$(sha256sum "$ROOT/runlogs/freeze/protocol.txt" | awk '{print $1}')
test "$actual_protocol" = "$expected_protocol"
test "$(sha256sum "$ROOT/prepared_data/reserved256.json" | awk '{print $1}')" = 412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241
test "$(sha256sum "$ROOT/data/r2r_val_tiny.parquet" | awk '{print $1}')" = c4d2490a46148d7801b123c4c007ce811a2c1a43f828412357702acd0a9e9bb0
"$BASE/activevln_train_env/bin/python" "$STATE/audit_remax_training64.py" --root "$ROOT" --output "$STATE/training_combined_audit64.json"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors.index.json"
python - "$CHECKPOINT" "$TRAIN/checkpoint_identity.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); out=Path(sys.argv[2])
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
files=sorted(x for x in p.iterdir() if x.is_file())
idx=json.loads((p/"model.safetensors.index.json").read_text())
weight_files={f.name:{"bytes":f.stat().st_size} for f in files if f.name.endswith(".safetensors")}
assert sum(x["bytes"] for x in weight_files.values())==idx["metadata"]["total_size"]
out.write_text(json.dumps({"schema":"remax_final_checkpoint_metadata_v1","checkpoint_step":64,
 "config_sha256":sha(p/"config.json"),"index_sha256":sha(p/"model.safetensors.index.json"),
 "weight_files":weight_files,"index_total_size":idx["metadata"]["total_size"],
 "train_log_sha256":sha(Path(sys.argv[2]).parent/"train.log")},indent=2)+"\n")
print("CHECKPOINT_METADATA_PASS",len(files))
PY
if curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then
  test -s "$ROOT/runlogs/service/server.pid"
  service_pid=$(cat "$ROOT/runlogs/service/server.pid")
  ps -o args= -p "$service_pid" | grep -F "server.port=5087" >/dev/null
  kill -TERM "$service_pid"
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then break; fi
    sleep 2
  done
  if curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then echo "owned Habitat server did not stop" >&2; exit 1; fi
fi
for _ in $(seq 1 180); do
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  m2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$m0" -lt 10000 && test "$m1" -lt 10000 && test "$m2" -lt 10000; then break; fi
  sleep 10
done
test "$m0" -lt 10000 && test "$m1" -lt 10000 && test "$m2" -lt 10000
test -f "$SFTROOT/positive_initial_sft.completed"
test ! -f "$SFTROOT/positive_initial_sft.failed"
SFTSTATS="$SFTROOT/positive_initial_sft"
(cd "$SFTSTATS"; sha256sum -c "$ROOT/runlogs/freeze/sft_reserved_raw_stats.sha256" >/dev/null)
if test -L "$RESULT/positive_initial_sft"; then
  test "$(readlink "$RESULT/positive_initial_sft")" = "$SFTSTATS"
elif test ! -e "$RESULT/positive_initial_sft"; then
  ln -s "$SFTSTATS" "$RESULT/positive_initial_sft"
else
  echo "refusing unexpected existing SFT result path" >&2; exit 1
fi
if test ! -e "$RESULT/positive_initial_sft.completed"; then ln -s "$SFTROOT/positive_initial_sft.completed" "$RESULT/positive_initial_sft.completed"; fi
"$BASE/activevln_train_env/bin/python" tools/validate_train_label.py --root "$RESULT" \
  --manifest prepared_data/reserved256.json --role reserved --label positive_initial_sft \
  --output "$RESULT/positive_initial_sft.validated.json" >"$STATE/sft_raw_validation.log"
"$BASE/activevln_train_env/bin/python" tools/verify_positive_engine_dtype.py \
  --log "$SFTROOT/vllm_positive_initial_sft.log" \
  --checkpoint "$BASE/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
  --expected-dtype float16 --output "$RESULT/positive_initial_sft.dtype.json" >"$STATE/sft_dtype_validation.log"
"$BASE/activevln_train_env/bin/python" - "$RESULT/positive_initial_sft.validated.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x["episodes"]==256 and x["successes"]==91 and x["inference_errors"]==0
assert x["manifest_sha256"]=="412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241"
assert math.isclose(x["spl"],0.3443758508038736,rel_tol=0,abs_tol=1e-10)
print("FP16_SFT_REFERENCE_RAW_VALIDATION_PASS")
PY
test -f "$CHECKPOINT/config.json"
test ! -e "$RESULT/$LABEL.completed"
test ! -e "$RESULT/$LABEL.failed"
if ss -ltn "( sport = :$PORT )" | grep -q LISTEN; then echo "evaluation port occupied" >&2; exit 1; fi
exec 8>"$ROOT/runlogs/gpu_eval_locks/gpu0.lock"
flock 8
memory=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$memory" -lt 10000
"$BASE/activevln_train_env/bin/python" "$STATE/verify_remax_eval_freeze.py" "$ROOT" "$STATE/evaluation_identity.json" "$STATE/evaluation_identity.sha256" "$STATE/complete_remax_after_train_eval_v2.sh" >"$STATE/evaluation_freeze_before_inference.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$STATE/reserved_candidate.opened"
export PYTHONPATH="$ROOT/vlnce_server:$ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/tools/vllm_compat:$PYTHONPATH" \
  "$BASE/activevln_train_env/bin/vllm" serve "$CHECKPOINT" --port "$PORT" \
  --dtype half --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  --disable-log-requests --seed 11 >"$RESULT/vllm_$LABEL.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
test "$ready" -eq 1
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 "$BASE/activevln_server_env/bin/python" tools/eval_train_scene_subset.py \
    --model-label "$LABEL" --manifest prepared_data/reserved256.json --result-root "$RESULT" \
    --role reserved --count 256 --shard-count 4 --shard-index "$shard" \
    --max-turns 12 --base-url "http://127.0.0.1:$PORT/v1" \
    >"$RESULT/eval_$LABEL"_shard"$shard".log 2>&1 &
  worker_pids+=("$!")
done
rc=0
for pid in "${worker_pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0
"$BASE/activevln_train_env/bin/python" tools/validate_train_label.py --root "$RESULT" \
  --manifest prepared_data/reserved256.json --role reserved --label "$LABEL" \
  --output "$RESULT/$LABEL.validated.json" >"$RESULT/$LABEL.validation.log"
test "$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["episodes"])' "$RESULT/$LABEL.validated.json")" = 256
kill -TERM "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""
"$BASE/activevln_train_env/bin/python" tools/verify_positive_engine_dtype.py \
  --log "$RESULT/vllm_$LABEL.log" --checkpoint "$CHECKPOINT" --expected-dtype float16 \
  --output "$RESULT/$LABEL.dtype.json" >"$RESULT/$LABEL.dtype_validation.log"
test -f "$RESULT/$LABEL.validated.json"
test ! -e "$RESULT/$LABEL.failed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$RESULT/$LABEL.completed"
"$BASE/activevln_train_env/bin/python" tools/analyze_train_scene_pair.py \
  --root "$RESULT" --manifest prepared_data/reserved256.json --role reserved \
  --control positive_initial_sft --candidate "$LABEL" \
  --compact "$STATE/remax_vs_sft_episodes.jsonl" --output "$STATE/remax_vs_sft_report.json" \
  >"$STATE/remax_vs_sft_analysis.log"
"$BASE/activevln_train_env/bin/python" tools/verify_positive_compact.py \
  --manifest prepared_data/reserved256.json --compact "$STATE/remax_vs_sft_episodes.jsonl" \
  --report "$STATE/remax_vs_sft_report.json" --validators "$RESULT" \
  --output "$STATE/remax_vs_sft_independent_recount.json" >"$STATE/remax_vs_sft_independent.log"
"$BASE/activevln_train_env/bin/python" - "$STATE/remax_vs_sft_report.json" "$STATE/gate.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); sr=float(x["paired_sr_points"]); spl=float(x["paired_spl_points"])
g={"schema":"remax_reserved_screen_gate_v1","role":"reserved","manifest_sha256":x["manifest_sha256"],
 "candidate":x["candidate"],"control":x["control"],"episodes":x["episodes"],"scenes":x["scenes"],
 "paired_sr_points":sr,"paired_spl_points":spl,"threshold_each_points":2.0,
 "pass":bool(sr>=2.0 and spl>=2.0),"baseline_previously_evaluated":True,"clean_generalization_test":False}
with open(sys.argv[2],"w") as f: json.dump(g,f,indent=2); f.write("\n")
print(json.dumps(g))
PY
(cd "$ROOT"; sha256sum -c runlogs/freeze/source_files.sha256 >/dev/null)
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$STATE/suite.completed"

