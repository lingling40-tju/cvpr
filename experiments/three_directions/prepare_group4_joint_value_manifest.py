"""Freeze expert same-start instruction contrasts on the group-four split.

The records and RGB frames already exist. Labels are selected only from
R2R-train metadata and the previously audited expert collection. This
script never reads model scores or val-unseen outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pose(row: dict) -> tuple:
    return (str(row["scene_id"]),
            tuple(round(float(v), 4) for v in row["start_position"]),
            tuple(round(float(v), 4) for v in row["start_rotation"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    split = json.loads(args.scene_split.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    labels = json.loads(args.expert_labels.read_text())
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            split["schema"] != "stop_history_lora_scene_split_v1" or \
            expert["schema"] != "stop_history_expert_manifest_v1" or \
            labels["schema"] != "stop_history_train_only_label_audit_v1" or \
            split["label_audit_sha256"] != digest(args.expert_labels) or \
            labels["manifest_sha256"] != digest(args.expert_manifest) or \
            group["train_dataset_sha256"] != digest(args.train_dataset) or \
            expert["source_sha256"]["train_dataset"] != digest(args.train_dataset):
        raise ValueError("frozen source provenance mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episode_list = json.load(stream)["episodes"]
    episodes = {str(row["episode_id"]): row for row in episode_list}
    if len(episodes) != len(episode_list):
        raise ValueError("duplicate train episode ID")
    original = {}
    for old_part, rows in expert["selected"].items():
        for row in rows:
            eid = str(row["episode_id"])
            if eid in original:
                raise ValueError(f"duplicate expert selection {eid}")
            original[eid] = (old_part, row)
    label_by_id = {str(row["episode_id"]): row
                   for rows in labels["labels"].values() for row in rows}
    if len(label_by_id) != sum(len(rows) for rows in labels["labels"].values()):
        raise ValueError("duplicate expert label")
    selected = {}
    inventory = {}
    scenes_seen = set()
    for part in ("fit", "development", "audit"):
        group_scenes = {row["scene_id"] for row in group["selected"][part]}
        if group_scenes != set(split["scene_split"][part]) or \
                scenes_seen & group_scenes:
            raise ValueError(f"group-four scene split mismatch {part}")
        scenes_seen.update(group_scenes)
        rows = []
        for source in split["selected"][part]:
            eid = str(source["episode_id"])
            label = label_by_id[eid]
            old_part, plan = original[eid]
            if not source["safe_wrong_instruction"]:
                continue
            if not (label["within_12_turns"] and label["stop_labels_valid"] and
                    label["safe_wrong_instruction"]):
                raise ValueError(f"invalid same-start expert label {eid}")
            correct, wrong = episodes[eid], episodes[str(plan["swap_episode_id"])]
            if pose(correct) != pose(wrong) or \
                    math.dist(correct["goals"][0]["position"],
                              wrong["goals"][0]["position"]) < 3.5:
                raise ValueError(f"same-start/different-goal mismatch {eid}")
            path = args.expert_root / old_part / "records" / f"{eid}.json"
            record = json.loads(path.read_text())
            if record["manifest_sha256"] != digest(args.expert_manifest) or \
                    record["episode_id"] != eid or \
                    record["scene_id"] != source["scene_id"] or \
                    record["trajectory_id"] != source["trajectory_id"] or \
                    record["turn_count"] > 12 or \
                    record["instruction"].strip() != plan["instruction"].strip() or \
                    record["wrong_instruction"].strip() != plan["wrong_instruction"].strip():
                raise ValueError(f"expert history mismatch {eid}")
            frames = [record["initial_image"]] + [turn["image"] for turn in record["turns"]]
            if not all((args.expert_root / old_part / frame).is_file() for frame in frames):
                raise ValueError(f"missing expert RGB frame {eid}")
            rows.append({"episode_id": eid, "scene_id": record["scene_id"],
                         "trajectory_id": record["trajectory_id"],
                         "source_part": old_part, "turn_count": record["turn_count"],
                         "wrong_episode_id": str(plan["swap_episode_id"]),
                         "record_sha256": digest(path)})
        rows.sort(key=lambda row: row["episode_id"])
        if len({row["episode_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate expert in {part}")
        selected[part] = rows
        inventory[part] = {"expert_contrasts": len(rows),
                           "scenes": len({row["scene_id"] for row in rows}),
                           "original_source_parts": dict(Counter(
                               row["source_part"] for row in rows))}
    expected = {"fit": 596, "development": 161, "audit": 123}
    if {part: x["expert_contrasts"] for part, x in inventory.items()} != expected:
        raise ValueError(f"unexpected expert contrast coverage: {inventory}")
    result = {"schema": "group4_joint_value_expert_manifest_v1",
              "selection": "existing audited <=12-turn expert histories with exact same-start natural different-goal instruction, scene partition shared with complete four-rollout cache",
              "source_sha256": {
                  "group_manifest": digest(args.group_manifest),
                  "scene_split": digest(args.scene_split),
                  "expert_manifest": digest(args.expert_manifest),
                  "expert_labels": digest(args.expert_labels),
                  "train_dataset": digest(args.train_dataset)},
              "inventory": inventory, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
