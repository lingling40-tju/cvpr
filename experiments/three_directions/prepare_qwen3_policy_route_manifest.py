"""Select exact-same-start wrong-goal policy histories without model scores.

The output contains only source IDs, hashes, and outcome labels. Natural
instruction text and replay RGB remain on the licensed experiment host.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    group_sha = digest(args.group_manifest)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            group["train_dataset_sha256"] != digest(args.train_dataset):
        raise ValueError("group or dataset source mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate R2R train ID")
    by_pose = defaultdict(list)
    for row in episodes:
        by_pose[pose(row)].append(row)
    selected = {}
    inventory = {}
    used_scenes = set()
    for part in ("fit", "development", "audit"):
        grouped = defaultdict(list)
        for plan in group["selected"][part]:
            grouped[(plan["seed"], str(plan["episode_id"]))].append(plan)
        entries = []
        group_sizes = Counter()
        for (seed, eid), plans in sorted(grouped.items()):
            if len(plans) != 4 or {plan["variant"] for plan in plans} != {0, 1, 2, 3}:
                raise ValueError("incomplete group-four rollouts")
            original = by_id[eid]
            candidates = []
            for other in by_pose[pose(original)]:
                wrong_id = str(other["episode_id"])
                if wrong_id == eid or \
                        other["instruction"]["instruction_text"].strip() == \
                        original["instruction"]["instruction_text"].strip():
                    continue
                goal_gap = math.dist(original["goals"][0]["position"],
                                     other["goals"][0]["position"])
                if goal_gap < 4:
                    continue
                rank = hashlib.sha256(
                    f"qwen3-policy-exact-goal-v1:{eid}:{wrong_id}".encode()).hexdigest()
                candidates.append((rank, wrong_id, goal_gap))
            if not candidates:
                continue
            _, wrong_id, gap = min(candidates)
            routes = []
            for plan in sorted(plans, key=lambda x: x["variant"]):
                record_id = f"s{seed}_e{eid}_v{plan['variant']}"
                path = args.turn_root / part / "records" / f"{record_id}.json"
                record = json.loads(path.read_text())
                if record["manifest_sha256"] != group_sha or \
                        record["record_id"] != record_id or \
                        str(record["episode_id"]) != eid or \
                        record["scene_id"] != plan["scene_id"] or \
                        record["instruction"].strip() != original["instruction"]["instruction_text"].strip() or \
                        record["terminal_mode"] != plan["terminal_mode"]:
                    raise ValueError(f"replay identity mismatch {record_id}")
                anchors = [t for t in (3, 6) if len(record["turns"]) > t]
                frames = [record["initial_image"]] + [turn["image"] for turn in record["turns"]]
                if anchors and not all((args.turn_root / part / frame).is_file()
                                       for frame in frames[:max(anchors) + 1]):
                    raise ValueError(f"missing policy RGB {record_id}")
                routes.append({"record_id": record_id, "variant": plan["variant"],
                               "record_sha256": digest(path), "turns": len(record["turns"]),
                               "anchors": anchors,
                               "success_for_analysis_only":
                                   plan["terminal_mode"] == "successfully reached the goal."})
            entries.append({"seed": seed, "episode_id": eid, "scene_id": plans[0]["scene_id"],
                            "wrong_episode_id": wrong_id,
                            "goal_gap_m_for_selection_only": gap, "routes": routes})
            group_sizes[len(routes)] += 1
        scenes = {entry["scene_id"] for entry in entries}
        if used_scenes & scenes:
            raise ValueError("policy part scene overlap")
        used_scenes.update(scenes)
        selected[part] = entries
        inventory[part] = {"groups": len(entries), "rollouts": 4 * len(entries),
                           "preterminal_t3": sum(3 in route["anchors"] for e in entries for route in e["routes"]),
                           "preterminal_t6": sum(6 in route["anchors"] for e in entries for route in e["routes"]),
                           "success_t6": sum(6 in r["anchors"] and r["success_for_analysis_only"]
                                             for e in entries for r in e["routes"]),
                           "mixed_groups_t6": sum(0 < sum(6 in r["anchors"] and r["success_for_analysis_only"]
                                                             for r in e["routes"]) <
                                                   sum(6 in r["anchors"] for r in e["routes"])
                                                   for e in entries),
                           "scenes": len(scenes), "group_size_histogram": dict(group_sizes)}
        inventory[part]["success_failure_pairs_t6"] = sum(
            sum(6 in r["anchors"] and r["success_for_analysis_only"] for r in e["routes"]) *
            sum(6 in r["anchors"] and not r["success_for_analysis_only"] for r in e["routes"])
            for e in entries)
    result = {"schema": "qwen3_policy_route_manifest_v1",
              "selection": "exact same scene/start position/start rotation, different instruction and goal >=4m; SHA-ranked natural wrong goal; complete group-four rollouts; preterminal anchors 3 and 6 only",
              "source_sha256": {"group_manifest": group_sha,
                                "train_dataset": digest(args.train_dataset)},
              "inventory": inventory, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory, "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
