#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root=$base/ActiveVLN_lora_group4_20261012
sft=$base/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_development
grpo=$base/ActiveVLN_three_direction_20261009/runlogs/development256
state=$root/runlogs/lora_rloo_v5_development_eval
result=$root/runlogs/lora_rloo_v5_development256
manifest=$root/prepared_data/development256.json
identity=$root/runlogs/lora_rloo_v5_development_eval_identity.json
serving=$root/runlogs/lora_rloo_v5_development_serving_identity.json
label=lora_rloo_qv_rank16_n4_seed11_v5_global64
base_model=$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn
adapter=$root/verl_checkpoints/lora_rloo_qv_rank16_n4_seed11_v5/global_step_64/actor/lora_adapter
port=8161
mkdir -p "$state" "$result" "$root/runlogs/gpu_eval_locks"
exec 8>"$state/eval.lock"; flock -n 8 || exit 2
exec 9>"$root/runlogs/gpu_eval_locks/gpu0.lock"; flock -n 9 || exit 2
test ! -f "$state/suite.failed"
test ! -f "$state/suite.completed" || exit 0
train=$root/runlogs/lora_rloo_qv_rank16_n4_seed11_v5_continue64
test -f "$train/completed"; test ! -f "$train/failed"
test -s "$adapter/adapter_model.safetensors"; test -f "$adapter/adapter_config.json"
test "$(sha256sum "$identity" | awk '{print $1}')" = d5c846557bb9ae827fc32d5ac4f33d0a4fd391937a519bd5189ba193727874f6
test "$(sha256sum "$serving" | awk '{print $1}')" = f4f7620df119fb43136c0440187a5377ddf26e2182281e3a2674d1e716447b91
"$base/activevln_server_env/bin/python" - "$identity" <<'PY'
import hashlib,json,sys
for name,item in json.load(open(sys.argv[1]))["frozen_files"].items():
 if hashlib.sha256(open(item["path"],"rb").read()).hexdigest()!=item["sha256"]: raise SystemExit("frozen hash mismatch: "+name)
print("frozen inputs verified")
PY
test "$(sha256sum "$manifest" | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
test -f "$sft/positive_initial_sft.validated.json"; test -f "$sft/positive_initial_sft.dtype.json"; test -f "$sft/positive_initial_sft.completed"
test -f "$grpo/td_grpo_anchor_n4_seed11.validated.json"; test -f "$grpo/td_grpo_anchor_n4_seed11.completed"
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 10000
! ss -ltn "( sport = :$port )" | grep -q LISTEN
curl -fsS --max-time 3 http://127.0.0.1:5091/health >/dev/null
ln -sfn "$manifest" "$result/manifest.json"
ln -sfn "$sft/positive_initial_sft" "$result/positive_initial_sft"
ln -sfn "$sft/positive_initial_sft.validated.json" "$result/positive_initial_sft.validated.json"
ln -sfn "$sft/positive_initial_sft.completed" "$result/positive_initial_sft.completed"
ln -sfn "$grpo/td_grpo_anchor_n4_seed11" "$result/td_grpo_anchor_n4_seed11"
ln -sfn "$grpo/td_grpo_anchor_n4_seed11.validated.json" "$result/td_grpo_anchor_n4_seed11.validated.json"
ln -sfn "$grpo/td_grpo_anchor_n4_seed11.completed" "$result/td_grpo_anchor_n4_seed11.completed"
printf '%s\n' "$$" >"$state/launcher.pid"
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root"
server_pid=; evalpid0=; evalpid1=; evalpid2=; evalpid3=
cleanup() {
 rc=$?
 for pid in "$evalpid0" "$evalpid1" "$evalpid2" "$evalpid3"; do test -z "$pid" || kill "$pid" 2>/dev/null || true; done
 if test -n "$server_pid"; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
 if test "$rc" -ne 0; then echo "$rc" >"$state/suite.failed"; fi
}
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$root/tools/vllm_compat:$root/vlnce_server:$root" \
 "$base/activevln_train_env/bin/vllm" serve "$base_model" --port "$port" --dtype half \
 --enable-lora --max-lora-rank 16 --max-loras 1 \
 --lora-modules "lora_rloo_qv_rank16_n4_seed11_v5_global64=$adapter" \
 --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
 --disable-log-requests --seed 11 >"$state/vllm_candidate_retry1.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120); do
 if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >"$state/models.json" && grep -Fq "$label" "$state/models.json"; then ready=1; break; fi
 kill -0 "$server_pid" 2>/dev/null || break; sleep 3
done
test "$ready" -eq 1
"$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py --log "$state/vllm_candidate_retry1.log" --checkpoint "$base_model" --expected-dtype float16 --output "$state/candidate_actual_dtype_retry1.json"
grep -Fq "$adapter" "$state/vllm_candidate_retry1.log"
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$root/vlnce_server:$root" "$base/activevln_server_env/bin/python" tools/eval_lora_candidate.py --model-label "$label" --served-model-name "$label" --manifest "$manifest" --result-root "$result" --role development --count 256 --shard-count 4 --shard-index 0 --max-turns 12 --base-url "http://127.0.0.1:$port/v1" >"$result/eval_"$label"_shard0.log" 2>&1 & evalpid0=$!
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$root/vlnce_server:$root" "$base/activevln_server_env/bin/python" tools/eval_lora_candidate.py --model-label "$label" --served-model-name "$label" --manifest "$manifest" --result-root "$result" --role development --count 256 --shard-count 4 --shard-index 1 --max-turns 12 --base-url "http://127.0.0.1:$port/v1" >"$result/eval_"$label"_shard1.log" 2>&1 & evalpid1=$!
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$root/vlnce_server:$root" "$base/activevln_server_env/bin/python" tools/eval_lora_candidate.py --model-label "$label" --served-model-name "$label" --manifest "$manifest" --result-root "$result" --role development --count 256 --shard-count 4 --shard-index 2 --max-turns 12 --base-url "http://127.0.0.1:$port/v1" >"$result/eval_"$label"_shard2.log" 2>&1 & evalpid2=$!
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$root/vlnce_server:$root" "$base/activevln_server_env/bin/python" tools/eval_lora_candidate.py --model-label "$label" --served-model-name "$label" --manifest "$manifest" --result-root "$result" --role development --count 256 --shard-count 4 --shard-index 3 --max-turns 12 --base-url "http://127.0.0.1:$port/v1" >"$result/eval_"$label"_shard3.log" 2>&1 & evalpid3=$!
rc=0
wait "$evalpid0" || rc=1
wait "$evalpid1" || rc=1
wait "$evalpid2" || rc=1
wait "$evalpid3" || rc=1
test "$rc" -eq 0
"$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest "$manifest" --role development --label "$label" --output "$result/$label.validated.json" >"$result/$label.validation.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/$label.completed"
for pair in sft grpo; do
 if test "$pair" = sft; then comparator=positive_initial_sft; else comparator=td_grpo_anchor_n4_seed11; fi
 compact="$state/"$pair"_paired_episodes.jsonl"; report="$state/"$pair"_paired_report.json"
 "$base/activevln_server_env/bin/python" tools/analyze_train_scene_pair.py --root "$result" --manifest "$manifest" --role development --control "$comparator" --candidate "$label" --compact "$compact" --output "$report" >"$state/"$pair"_analysis.log"
 "$base/activevln_server_env/bin/python" tools/verify_positive_compact.py --manifest "$manifest" --compact "$compact" --report "$report" --validators "$result" --output "$state/"$pair"_independent_recount.json" >"$state/"$pair"_independent_recount.log"
done
"$base/activevln_server_env/bin/python" - "$identity" "$state" <<'PY'
import json,sys
i=json.load(open(sys.argv[1])); s=sys.argv[2]
r={k:json.load(open(s+"/"+k+"_paired_report.json")) for k in ("sft","grpo")}
g={k:{"pass":v["paired_sr_points"]>=2 and v["paired_spl_points"]>=2,"sr_pp":v["paired_sr_points"],"spl_pp":v["paired_spl_points"]} for k,v in r.items()}
o={"schema":"lora_rloo_v5_development_suite_v1","freeze_sha256":"d5c846557bb9ae827fc32d5ac4f33d0a4fd391937a519bd5189ba193727874f6","screen":i["screen"],"candidate":"lora_rloo_qv_rank16_n4_seed11_v5_global64","comparisons":r,"gates":g,"scale_eligible":all(x["pass"] for x in g.values()),"interpretation":"Adaptive R2R-train development screen; one seed; not a clean test or deployable semantic-reward validation."}
json.dump(o,open(s+"/three_comparator_summary.json","w"),indent=2); open(s+"/three_comparator_summary.json","a").write("\n")
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
