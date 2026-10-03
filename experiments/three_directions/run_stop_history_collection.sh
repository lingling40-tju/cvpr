#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_three_directions_20261002"
run="$root/runlogs/ordinal_progress/stop_history_collection"
manifest="$root/runlogs/ordinal_progress/stop_history_expert_manifest.json"
output="$root/runlogs/ordinal_progress/stop_history_expert_frames"
expected_sha=3d8ab23313377729501a0dee4a9274231fb927f73033809a68a08fdcf22e1e43
mkdir -p "$run" "$output"
exec 9>"$run/collection.lock"
flock -n 9 || { echo 'stop-history collection already running' >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_sha"
if test -f "$run/completed"; then exit 0; fi
rm -f "$run/failed"
on_exit() {
  status=$?
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$run/failed"; fi
}
trap on_exit EXIT
cd "$root"
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
python="$base/activevln_server_env/bin/python"

collect() {
  local part=$1 gpu=$2
  if test -f "$output/$part/summary.json" && \
      "$python" - "$output/$part/summary.json" "$manifest" "$part" <<'PY'
import hashlib,json,sys
summary=json.load(open(sys.argv[1]))
source=open(sys.argv[2],"rb").read()
manifest=json.loads(source)
part=sys.argv[3]
expected={str(row["episode_id"]) for row in manifest["selected"][part]}
assert summary["manifest_sha256"]==hashlib.sha256(source).hexdigest()
assert summary["requested"]==summary["completed"]==len(expected)
assert set(map(str,summary["completed_episode_ids"]))==expected
assert summary["smoke_limit"]==0 and not summary["errors"]
PY
  then
    echo "verified existing $part collection" >"$run/$part.log"
    return 0
  fi
  "$python" tools/collect_stop_history_expert.py \
    --manifest "$manifest" --part "$part" \
    --output-root "$output" --gpu "$gpu" \
    >"$run/$part.log" 2>&1
}

collect fit 0 & fit_pid=$!; printf '%s\n' "$fit_pid" >"$run/fit.pid"
collect development 1 & development_pid=$!; printf '%s\n' "$development_pid" >"$run/development.pid"
collect audit 2 & audit_pid=$!; printf '%s\n' "$audit_pid" >"$run/audit.pid"
status=0
wait "$fit_pid" || status=1
wait "$development_pid" || status=1
wait "$audit_pid" || status=1
test "$status" -eq 0

"$python" - "$manifest" "$output" "$run/collection_audit.json" <<'PY'
import hashlib, json, sys
from collections import Counter
from pathlib import Path

manifest_path, output, audit_path = map(Path, sys.argv[1:])
source = manifest_path.read_bytes()
manifest = json.loads(source)
sha = hashlib.sha256(source).hexdigest()
result = {"schema": "stop_history_collection_audit_v1",
          "manifest_sha256": sha, "parts": {}}
all_keys = set()
for part, plans in manifest["selected"].items():
    folder = output / part
    summary = json.loads((folder / "summary.json").read_text())
    ids = {str(plan["episode_id"]) for plan in plans}
    if (summary["manifest_sha256"] != sha or summary["requested"] != len(ids)
            or summary["completed"] != len(ids) or summary["errors"]
            or summary["smoke_limit"] != 0
            or set(map(str, summary["completed_episode_ids"])) != ids):
        raise ValueError(f"invalid {part} coverage")
    labels = Counter()
    scenes = set()
    for plan in plans:
        eid = str(plan["episode_id"])
        record = json.loads((folder / "records" / f"{eid}.json").read_text())
        key = (record["scene_id"], record["trajectory_id"])
        if key in all_keys or record["manifest_sha256"] != sha or \
                record["episode_id"] != eid or \
                record["scene_id"] != plan["scene_id"] or \
                record["instruction"] != plan["instruction"].strip():
            raise ValueError(f"record mismatch or trajectory leakage: {part}/{eid}")
        all_keys.add(key)
        scenes.add(record["scene_id"])
        if record["start_distance_to_goal_for_label_only"] >= 3.5:
            labels["far_start"] += 1
        if record["end_within_3m"]:
            labels["successful_end"] += 1
        if record["turn_count"] <= 12:
            labels["at_most_12_turns"] += 1
        for image in [record["initial_image"]] + [turn["image"] for turn in record["turns"]]:
            if not (folder / image).is_file():
                raise ValueError(f"missing image: {part}/{eid}/{image}")
    result["parts"][part] = {"episodes": len(ids), "scenes": len(scenes),
                             "labels": dict(labels)}
if result["parts"]["audit"]["labels"].get("successful_end", 0) < 100 or \
        result["parts"]["audit"]["labels"].get("far_start", 0) < 100:
    raise ValueError("audit STOP sample gate underpowered")
audit_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/completed"
