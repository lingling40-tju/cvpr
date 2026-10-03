"""Freeze expert 3-to-6 progress intervals with natural wrong goals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.expert_manifest.read_text())
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("wrong expert manifest")
    selected = {}
    inventory = {}
    all_scenes = set()
    for part in ("fit", "development"):
        rows = []
        for source in manifest["selected"][part]:
            if source["turn_count"] <= 6:
                continue
            eid = str(source["episode_id"])
            path = args.expert_root / source["source_part"] / "records" / f"{eid}.json"
            if digest(path) != source["record_sha256"]:
                raise ValueError(f"changed expert source {part}/{eid}")
            record = json.loads(path.read_text())
            if record["episode_id"] != eid or \
                    record["scene_id"] != source["scene_id"] or \
                    record["trajectory_id"] != source["trajectory_id"] or \
                    record["turn_count"] != source["turn_count"] or \
                    record["wrong_instruction_episode_id"] != source["wrong_episode_id"]:
                raise ValueError(f"invalid expert interval {part}/{eid}")
            before = record["turns"][2]["distance_to_goal_for_label_only"]
            after = record["turns"][5]["distance_to_goal_for_label_only"]
            if before - after < 1.0:
                continue
            root = args.expert_root / source["source_part"]
            for turn in record["turns"][:6]:
                if not (root / turn["image"]).is_file():
                    raise ValueError(f"missing expert frame {part}/{eid}")
            rows.append({"episode_id": eid,
                         "scene_id": source["scene_id"],
                         "source_part": source["source_part"],
                         "record_sha256": source["record_sha256"],
                         "early_anchor": 3, "late_anchor": 6})
        rows.sort(key=lambda row: row["episode_id"])
        scenes = {row["scene_id"] for row in rows}
        expected = {"fit": (503, 38), "development": (143, 8)}[part]
        if (len(rows), len(scenes)) != expected or scenes & all_scenes or \
                len({row["episode_id"] for row in rows}) != len(rows):
            raise ValueError(f"temporal pair coverage/leakage {part}")
        all_scenes.update(scenes)
        selected[part] = rows
        inventory[part] = {"temporal_interactions": len(rows),
                           "states_per_interaction": 4,
                           "scenes": len(scenes)}
    result = {"schema": "group4_temporal_interaction_manifest_v1",
              "selection": "previously audited same-start natural different-goal expert histories with at least seven motion turns and correct-goal geodesic decrease of at least one meter from turn 3 to turn 6; distance is label selection only, never model input",
              "source_expert_manifest_sha256": digest(args.expert_manifest),
              "inventory": inventory, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
