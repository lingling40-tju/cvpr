#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/stop_history_feature_cache"
manifest="$root/runlogs/ordinal_progress/stop_history_expert_manifest.json"
labels="$root/runlogs/ordinal_progress/stop_history_label_audit.json"
records="$root/runlogs/ordinal_progress/stop_history_expert_frames"
output="$root/runlogs/ordinal_progress/stop_history_features"
model="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
expected_manifest=3d8ab23313377729501a0dee4a9274231fb927f73033809a68a08fdcf22e1e43
mkdir -p "$run" "$output"
exec 9>"$run/cache.lock"
flock -n 9 || { echo 'stop-history feature cache already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_manifest"
test -f "$root/runlogs/ordinal_progress/stop_history_collection/completed"
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_train_env/bin/python"

cache_part() {
  local part=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/cache_stop_history_features.py \
    --manifest "$manifest" --label-audit "$labels" \
    --records-root "$records" --model "$model" \
    --part "$part" --output-root "$output" \
    >"$run/$part.log" 2>&1
}

cache_part fit 0 & fit_pid=$!; printf '%s\n' "$fit_pid" >"$run/fit.pid"
cache_part development 1 & development_pid=$!; printf '%s\n' "$development_pid" >"$run/development.pid"
cache_part audit 2 & audit_pid=$!; printf '%s\n' "$audit_pid" >"$run/audit.pid"
status=0
wait "$fit_pid" || status=1
wait "$development_pid" || status=1
wait "$audit_pid" || status=1
test "$status" -eq 0

"$python" - "$manifest" "$labels" "$output" "$run/feature_audit.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch

manifest_path,label_path,output,audit_path=map(Path,sys.argv[1:])
manifest_bytes=manifest_path.read_bytes()
manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
labels=json.loads(label_path.read_text())
result={"schema":"stop_history_feature_coverage_v1",
        "manifest_sha256":manifest_hash,"parts":{}}
for part in ("fit","development","audit"):
    folder=output/part
    summary=json.loads((folder/"summary.json").read_text())
    eligible={str(row["episode_id"]):row for row in labels["labels"][part]
              if row["within_12_turns"]}
    expected=set(eligible)
    if (summary["manifest_sha256"]!=manifest_hash or
            summary["requested"]!=len(expected) or
            summary["completed"]!=len(expected) or
            set(map(str,summary["completed_episode_ids"]))!=expected or
            summary["errors"] or summary["limit"]!=0):
        raise ValueError(f"incomplete {part} feature cache")
    safe=0
    maximum_context=0
    for eid,row in eligible.items():
        payload=torch.load(folder/"records"/f"{eid}.pt",
                           map_location="cpu",weights_only=True)
        if (payload["episode_id"]!=eid or
                payload["manifest_sha256"]!=manifest_hash or
                payload["hidden"].shape!=(3,2048) or
                payload["stop_margin"].shape!=(3,) or
                not torch.isfinite(payload["hidden"]).all() or
                not torch.isfinite(payload["stop_margin"]).all()):
            raise ValueError(f"invalid {part}/{eid} feature")
        maximum_context=max(maximum_context,int(payload["context_tokens"].max()))
        if row["safe_wrong_instruction"]:
            safe+=1
            if (payload["wrong_hidden"].shape!=(2048,) or
                    not torch.isfinite(payload["wrong_hidden"]).all() or
                    not __import__("math").isfinite(payload["wrong_stop_margin"])):
                raise ValueError(f"invalid {part}/{eid} wrong instruction")
    result["parts"][part]={"records":len(expected),
                           "safe_wrong_instruction_records":safe,
                           "max_context_tokens":maximum_context,
                           "extraction_elapsed_seconds":summary["elapsed_seconds"]}
if result["parts"]["audit"]["records"]<100 or \
        result["parts"]["audit"]["safe_wrong_instruction_records"]<100:
    raise ValueError("audit feature sample gate underpowered")
audit_path.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
