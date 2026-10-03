#!/usr/bin/env bash
set -euo pipefail

kind=${1:?expected initial or evidence}
case "$kind" in initial|evidence) ;; *) echo 'bad source kind' >&2; exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/group4_evidence_onset_scores/$kind"
mkdir -p "$run"
exec 9>"$run/score.lock"
flock -n 9 || { echo "scorer $kind already running" >&2; exit 2; }
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_train_env/bin/python"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
if test "$kind" = initial; then
  checkpoint=runlogs/ordinal_progress/history_grounding_lora_seed11/interim_selected.pt
else
  checkpoint=runlogs/ordinal_progress/group4_evidence_onset_lora_seed11/final_adapter.pt
fi
test -f "$checkpoint"
manifest=runlogs/ordinal_progress/group4_evidence_onset_manifest.json
test "$(sha256sum "$manifest" | awk '{print $1}')" = \
  41b465132819ae6060d06b3cf0a6ff964d9be83d3ace265ae39414aa1258f26f
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
score() {
  local shard=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/score_group4_evidence_onset_dev.py \
    --manifest "$manifest" \
    --expert-root runlogs/ordinal_progress/stop_history_expert_frames \
    --model "$model" --checkpoint "$checkpoint" --source-kind "$kind" \
    --frozen-weights runlogs/ordinal_progress/group4_joint_value_development.pt \
    --shard "$shard" --shards 3 --output "$run/shard${shard}.json" \
    >"$run/shard${shard}.log" 2>&1
}
score 0 1 & pid0=$!
score 1 2 & pid1=$!
score 2 3 & pid2=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
test "$status" -eq 0
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
