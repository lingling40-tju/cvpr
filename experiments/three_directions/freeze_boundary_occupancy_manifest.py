"""Separate RGB replay plans from privileged boundary labels."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from preflight_boundary_occupancy_source import DATA_SHA, digest


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_output == args.labels_output or digest(args.dataset) != DATA_SHA:
        raise ValueError("changed train data or colliding outputs")
    report = json.loads(args.preflight.read_text())
    if report.get("schema") != "boundary_occupancy_train_scene_source_preflight_v1" or \
            report.get("group_size") != 4 or \
            report.get("seeds") != [11, 22, 33] or \
            report.get("source_sha256", {}).get("dataset") != DATA_SHA or \
            report.get("enough_coverage_for_rgb_replay") is not True or \
            not all(report.get("coverage_gates", {}).values()):
        raise ValueError("frozen n4 source gate not passed")
    split = report["scene_split"]
    if set(split) != {"fit", "development", "audit"} or \
            len({scene for scenes in split.values() for scene in scenes}) != \
            sum(len(scenes) for scenes in split.values()):
        raise ValueError("scene leakage across partitions")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["episodes"]
    episodes = {str(row["episode_id"]): row for row in rows}
    if len(episodes) != len(rows):
        raise ValueError("duplicate natural episode")
    replay_selected = {}
    label_selected = {}
    seen = set()
    for part in ("fit", "development", "audit"):
        replay_selected[part] = []
        label_selected[part] = []
        for item in report["selected"][part]:
            rid, eid = item["record_id"], str(item["episode_id"])
            if rid in seen or eid not in episodes or \
                    item["scene_id"] not in split[part] or \
                    str(episodes[eid]["scene_id"]) != item["scene_id"] or \
                    item["seed"] not in (11, 22, 33) or \
                    not 0 <= item["variant"] < 4 or \
                    rid != f"s{item['seed']}_e{eid}_v{item['variant']}":
                raise ValueError(f"invalid selected record: {rid}")
            seen.add(rid)
            outside, inside = (item["outside_state_index"],
                               item["inside_state_index"])
            if not isinstance(outside, int) or not isinstance(inside, int) or \
                    not 0 <= outside < inside <= outside + 2 or \
                    not 3.5 <= item["outside_distance_m_for_label_only"] <= 4.5 or \
                    not 0 <= item["inside_distance_m_for_label_only"] <= 3.0:
                raise ValueError(f"broken boundary label: {rid}")
            wrong_id = (item["exact_start_far_wrong_instruction_episode_id"]
                        or item["far_wrong_instruction_episode_id"])
            true_instruction = episodes[eid]["instruction"]["instruction_text"].strip()
            wrong_instruction = (episodes[wrong_id]["instruction"]["instruction_text"].strip()
                                 if wrong_id else None)
            if not true_instruction or wrong_instruction == true_instruction:
                raise ValueError(f"missing instruction contrast: {rid}")
            replay_selected[part].append({
                "record_id": rid, "seed": item["seed"],
                "episode_id": eid, "variant": item["variant"],
                "scene_id": item["scene_id"],
                "instruction": true_instruction,
                "wrong_instruction": wrong_instruction,
                "wrong_instruction_same_start": bool(
                    item["exact_start_far_wrong_instruction_episode_id"]),
                "outside_state_index": outside,
                "inside_state_index": inside,
            })
            label_selected[part].append({
                "record_id": rid, "outside_distance_m":
                    item["outside_distance_m_for_label_only"],
                "inside_distance_m": item["inside_distance_m_for_label_only"],
                "task_success": item["task_success_for_audit_only"],
                "wrong_instruction_episode_id": wrong_id,
            })
        if len(replay_selected[part]) != report["counts"][part]["paired_boundary_rollouts"]:
            raise ValueError(f"part coverage mismatch: {part}")
    replay = {
        "schema": "boundary_occupancy_rgb_replay_manifest_v1",
        "group_size": 4, "seeds": [11, 22, 33],
        "preflight_sha256": digest(args.preflight),
        "source_sha256": report["source_sha256"],
        "scene_split": split, "selected": replay_selected,
        "interpretation": "Train-only RGB replay metadata; no geodesic labels in model input",
    }
    write_json(args.replay_output, replay)
    labels = {
        "schema": "boundary_occupancy_privileged_labels_v1",
        "group_size": 4, "preflight_sha256": digest(args.preflight),
        "replay_manifest_sha256": digest(args.replay_output),
        "selected": label_selected,
        "interpretation": "Simulator distances supervise and audit; never include this file in model input",
    }
    write_json(args.labels_output, labels)
    print(json.dumps({"replay_manifest_sha256": digest(args.replay_output),
                      "labels_sha256": digest(args.labels_output),
                      "counts": {part: len(items) for part, items in
                                 replay_selected.items()}}, indent=2))


if __name__ == "__main__":
    main()
