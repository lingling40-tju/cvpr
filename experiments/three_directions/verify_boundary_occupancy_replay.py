"""Fail-closed independent check of frozen RGB replay and label separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def check_part(part: str, plans: list[dict], labels: list[dict], root: Path,
               manifest_sha: str, shards: int, smoke_id: str) -> dict:
    selected = [p for p in plans if not smoke_id or p["record_id"] == smoke_id]
    if not selected or len({p["record_id"] for p in selected}) != len(selected):
        raise ValueError(f"missing or duplicate frozen plans: {part}")
    label_by_id = {x["record_id"]: x for x in labels}
    if len(label_by_id) != len(labels) or \
            {p["record_id"] for p in plans} != set(label_by_id):
        raise ValueError(f"label ID mismatch: {part}")
    summaries = [read(root / part / ("summary.json" if shards == 1 else
                 f"summary.shard{shard}.json")) for shard in range(shards)]
    if any(s["manifest_sha256"] != manifest_sha or s["part"] != part or
           s["shards"] != shards or s["shard"] != i or s["errors"] or
           s["completed"] != s["requested"] for i, s in enumerate(summaries)):
        raise ValueError(f"failed RGB replay shard: {part}")
    if sum(s["requested"] for s in summaries) != len(selected):
        raise ValueError(f"incomplete RGB replay: {part}")
    max_drift = 0.0
    for p in selected:
        rid = p["record_id"]
        r = read(root / part / "records" / f"{rid}.json")
        a = read(root / part / "audits" / f"{rid}.json")
        if r["schema"] != "boundary_occupancy_rgb_model_input_v1" or \
                a["schema"] != "boundary_occupancy_rgb_replay_audit_v1":
            raise ValueError(f"record schema or hash mismatch: {rid}")
        if r["manifest_sha256"] != manifest_sha or \
                a["manifest_sha256"] != manifest_sha or \
                r["record_id"] != a["record_id"] != rid:
            raise ValueError(f"record identity mismatch: {rid}")
        if (a["seed"], str(a["episode_id"]), a["variant"], a["scene_id"]) != \
                (p["seed"], str(p["episode_id"]), p["variant"], p["scene_id"]):
            raise ValueError(f"source identity mismatch: {rid}")
        inp = r["input"]
        if set(inp) != {"instruction", "wrong_instruction", "images",
                        "action_history_by_state"} or \
                inp["instruction"] != p["instruction"] or \
                inp["wrong_instruction"] != p["wrong_instruction"] or \
                set(inp["images"]) != {"outside", "inside"} or \
                set(inp["action_history_by_state"]) != {"outside", "inside"}:
            raise ValueError(f"model input leakage or mismatch: {rid}")
        if any(key in json.dumps(inp).lower() for key in
               ('oracle_distance', 'distance_to_goal', 'task_success')):
            raise ValueError(f"privileged field in model input: {rid}")
        distances = a["state_distance_m"]
        outside = str(p["outside_state_index"])
        inside = str(p["inside_state_index"])
        if set(distances) != {outside, inside}:
            raise ValueError(f"missing state audit: {rid}")
        label = label_by_id[rid]
        for key, reference, replay in (
                ("outside", label["outside_distance_m"], distances[outside]),
                ("inside", label["inside_distance_m"], distances[inside]),
                ("outside_source", label["outside_distance_m"],
                 a["source_outside_distance_m"]),
                ("inside_source", label["inside_distance_m"],
                 a["source_inside_distance_m"]),
                ("terminal", a["source_terminal_distance_m"],
                 a["terminal_distance_m"])):
            delta = abs(float(reference) - float(replay))
            if not math.isfinite(delta) or delta > .25:
                raise ValueError(f"{key} geodesic drift: {rid} {delta}")
            max_drift = max(max_drift, delta)
        if not 3.5 <= distances[outside] <= 4.75 or \
                not 0 <= distances[inside] <= 3.25:
            raise ValueError(f"invalid captured boundary state: {rid}")
        for state, index in (("outside", outside), ("inside", inside)):
            relative = Path(inp["images"][state])
            if relative.is_absolute() or ".." in relative.parts or \
                    relative.parts[:2] != ("frames", rid) or \
                    relative.name != f"{int(index):04d}.jpg":
                raise ValueError(f"unsafe or wrong image path: {rid}")
            with Image.open(root / part / relative) as frame:
                if frame.format != "JPEG" or not 1 <= frame.width <= 336 or \
                        not 1 <= frame.height <= 336:
                    raise ValueError(f"invalid RGB image: {rid}")
                frame.verify()
        for state in ("outside", "inside"):
            history = inp["action_history_by_state"][state]
            if any(set(t) != {"turn", "executed_actions"} or
                   "stop" in t["executed_actions"] for t in history):
                raise ValueError(f"invalid action history: {rid}")
        if (inp["action_history_by_state"]["outside"] !=
                inp["action_history_by_state"]["inside"][:len(
                    inp["action_history_by_state"]["outside"])]):
            raise ValueError(f"nonprefix action histories: {rid}")
    return {"part": part, "records": len(selected),
            "unique_episode_ids": len({p["episode_id"] for p in selected}),
            "rgb_images": len(selected) * 2, "max_geodesic_drift_m": max_drift,
            "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke-record-id", default="")
    args = parser.parse_args()
    manifest, labels, preflight = map(read, (args.manifest, args.labels,
                                            args.preflight))
    manifest_sha = sha(args.manifest)
    if manifest["schema"] != "boundary_occupancy_rgb_replay_manifest_v1" or \
            labels["schema"] != "boundary_occupancy_privileged_labels_v1" or \
            manifest["group_size"] != labels["group_size"] != 4 or \
            manifest["preflight_sha256"] != labels["preflight_sha256"] != sha(args.preflight) or \
            labels["replay_manifest_sha256"] != manifest_sha or \
            manifest["source_sha256"] != preflight["source_sha256"] or \
            not preflight["enough_coverage_for_rgb_replay"] or \
            not 1 <= args.shards <= 4:
        raise ValueError("changed or invalid frozen source")
    parts = ("fit",) if args.smoke_record_id else ("fit", "development")
    result = {"schema": "boundary_occupancy_rgb_replay_verification_v1",
              "manifest_sha256": manifest_sha,
              "smoke_record_id": args.smoke_record_id or None,
              "parts": [check_part(part, manifest["selected"][part],
                                   labels["selected"][part], args.replay_root,
                                   manifest_sha, args.shards,
                                   args.smoke_record_id) for part in parts],
              "navigation_result": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
