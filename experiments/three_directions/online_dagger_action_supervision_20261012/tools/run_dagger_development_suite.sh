#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_online_dagger_action_supervision_20261012"
pos="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/online_dagger_development256"
manifest="$root/prepared_data/development256.json"
sft="$pos/runlogs/positive_matched_precision_development"
sft_label=positive_initial_sft
mkdir -p "$state" "$root/runlogs/gpu_eval_locks"
exec 9>"$state/suite.lock"; flock -n 9 || { echo "development suite already active" >&2; exit 2; }
test -f "$root/runlogs/online_dagger_pilot64/suite.completed"
test ! -e "$state/suite.completed"
test ! -e "$state/suite.failed"
test "$(sha256sum "$manifest" | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
test "$(sha256sum "$pos/prepared_data/development256.json" | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
test ! -e "$state/results"
mkdir -p "$state/results"
cp "$manifest" "$state/results/manifest.json"
ln -s "$sft/$sft_label" "$state/results/$sft_label"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/results/$sft_label.completed"
sft_dtype="$sft/$sft_label.dtype.json"
sft_validation="$sft/$sft_label.validated.json"
test -f "$sft_dtype"; test -f "$sft_validation"
"$base/activevln_server_env/bin/python" "$root/tools/validate_train_label.py" \
 --root "$sft" --manifest "$manifest" --role development --label "$sft_label" \
 --output "$state/sft_reference_validated.json" >"$state/sft_reference_validation.log"
test ! -e "$state/evaluation_identity.json"
python3 - "$root" "$pos" "$state" "$manifest" "$sft_dtype" "$sft_validation" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,pos,state,manifest,sft_dtype,sft_valid=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ids=json.loads(manifest.read_text())
dtype=json.loads(sft_dtype.read_text())
assert dtype["actual_engine_dtype"]=="float16" and dtype["engine_seed"]==11
assert dtype["max_seq_len"]==16384 and dtype["tensor_parallel_size"]==1 and dtype["pipeline_parallel_size"]==1 and dtype["quantization"] is None
valid=json.loads(sft_valid.read_text())
assert valid["episodes"]==256 and valid["inference_errors"]==0
models={}
for arm in ("control","candidate"):
 label=f"online_dagger_{arm}_64step_seed11"
 ckpt=root/"verl_checkpoints"/label/"global_step_64"/"actor"/"huggingface"
 idx=json.loads((ckpt/"model.safetensors.index.json").read_text())
 models[label]={"checkpoint":str(ckpt),"config_sha256":sha(ckpt/"config.json"),
  "index_sha256":sha(ckpt/"model.safetensors.index.json"),
  "weight_files":{f.name:f.stat().st_size for f in sorted(ckpt.glob("model-*.safetensors"))},
  "tensor_count":len(idx["weight_map"])}
 assert models[label]["tensor_count"]>0 and len(models[label]["weight_files"])==4
files=["tools/eval_train_scene_subset.py","tools/validate_train_label.py","tools/analyze_train_scene_pair.py",
 "tools/run_dagger_development_model.sh","tools/verify_dagger_dev_raw.py","eval/vlnce/eval_vlnce.py"]
evaluator={f:sha(root/f) for f in files}
obj={"schema":"online_dagger_development256_freeze_v1","manifest_sha256":sha(manifest),
 "manifest_role":ids["role"],"episodes":len(ids["episode_ids"]),"scenes":len(set(ids["scene_ids"])),
 "sft_reference":{"label":"positive_initial_sft","dtype":"float16","dtype_proof_sha256":sha(sft_dtype),
  "validation_sha256":sha(sft_valid),"validated_before_model_calls_sha256":sha(state/"sft_reference_validated.json"),
  "completed_outputs_root":str(pos/"runlogs/positive_matched_precision_development/positive_initial_sft")},
 "trained_models":models,"source_sha256":evaluator,
 "decode":{"engine":"vLLM explicit half","engine_seed":11,"max_model_len":16384,
  "temperature":0.2,"top_p":0.8,"max_tokens":512,"max_turns":12,"max_pixels":76800,
  "parallel_arms":True,"habitat_shards_per_arm":4},
 "gate":"candidate vs matched FP16 SFT paired SR and SPL each >= +2.0 percentage points",
 "prohibited_splits":"reserved and val_unseen are not used"}
out=state/"evaluation_identity.json"; out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
print(json.dumps({"freeze":str(out),"sha256":sha(out),"episodes":256,"scenes":8,"models":len(models)}))
PY
# Keep the fixed Habitat server and all inference workers off GPU3.
busy=0
for gpu in 0 1 2; do
 mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
 if test "$mem" -ge 42000; then busy=1; fi
done
test "$busy" -eq 0
pids=()
cleanup(){ status=$?; for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi; }
trap cleanup EXIT
for arm in control candidate; do
 bash "$root/tools/run_dagger_development_model.sh" "$arm" >"$state/$arm.launcher.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
pids=()
test "$status" -eq 0
for label in online_dagger_control_64step_seed11 online_dagger_candidate_64step_seed11; do
 test -f "$state/results/$label.completed"; test ! -f "$state/results/$label.failed"
done
python="$base/activevln_server_env/bin/python"
"$python" "$root/tools/analyze_train_scene_pair.py" --root "$state/results" --manifest "$manifest" --role development \
 --control positive_initial_sft --candidate online_dagger_control_64step_seed11 \
 --compact "$state/control_vs_sft_episodes.jsonl" --output "$state/control_vs_sft.json" >"$state/control_vs_sft_analysis.log"
"$python" "$root/tools/analyze_train_scene_pair.py" --root "$state/results" --manifest "$manifest" --role development \
 --control positive_initial_sft --candidate online_dagger_candidate_64step_seed11 \
 --compact "$state/candidate_vs_sft_episodes.jsonl" --output "$state/candidate_vs_sft.json" >"$state/candidate_vs_sft_analysis.log"
"$python" "$root/tools/analyze_train_scene_pair.py" --root "$state/results" --manifest "$manifest" --role development \
 --control online_dagger_control_64step_seed11 --candidate online_dagger_candidate_64step_seed11 \
 --compact "$state/candidate_vs_control_episodes.jsonl" --output "$state/candidate_vs_control.json" >"$state/candidate_vs_control_analysis.log"
"$python" "$root/tools/verify_dagger_dev_raw.py" --root "$state/results" --sft-root "$sft" --manifest "$manifest" \
 --output "$state/independent_raw_recount.json" --compact "$state/independent_episodes.jsonl" >"$state/independent_raw_recount.log"
python3 - "$state" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
raw=json.loads((p/"independent_raw_recount.json").read_text())
checks=[("candidate_vs_sft",raw["candidate_vs_sft"]),("control_vs_sft",raw["control_vs_sft"]),("candidate_vs_control",raw["candidate_vs_control"])]
for name,expected in checks:
 got=json.loads((p/f"{name}.json").read_text())
 assert abs(got["paired_sr_points"]-expected["paired_sr_points"])<1e-9,(name,"SR")
 assert abs(got["paired_spl_points"]-expected["paired_spl_points"])<1e-9,(name,"SPL")
assert raw["episodes"]==256 and raw["scenes"]==8
print(json.dumps({"independent_raw_recount":"PASS","candidate_vs_sft":raw["candidate_vs_sft"],
 "candidate_vs_control":raw["candidate_vs_control"]}))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
