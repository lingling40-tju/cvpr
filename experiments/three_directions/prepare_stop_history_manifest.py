"""Freeze train-only expert trajectories for history/progress/STOP audits.

One natural instruction is selected per underlying trajectory. Development
and audit trajectories already used in policy-rollout supervision are
excluded. Selection is deterministic and independent of rendered images or
model scores. Wrong-instruction candidates share the exact start pose but
have a different goal at least 3.5 m away in Euclidean space.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


TARGET = {"fit": 896, "development": 160, "audit": 160}
SALT = "stop-history-expert-v1:"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def order(value: str) -> str:
    return hashlib.sha256((SALT + value).encode()).hexdigest()


def pose_key(row: dict) -> tuple:
    return (str(row["scene_id"]),
            tuple(round(float(v), 4) for v in row["start_position"]),
            tuple(round(float(v), 4) for v in row["start_rotation"]))


def goal(row: dict) -> tuple[float, ...]:
    return tuple(float(v) for v in row["goals"][0]["position"])


def goal_gap(a: dict, b: dict) -> float:
    return math.dist(goal(a), goal(b))


def eligible_gt(row: dict, gt: dict) -> bool:
    episode_id = str(row["episode_id"])
    if episode_id not in gt or float(row["info"]["geodesic_distance"]) < 4.0:
        return False
    actions = list(gt[episode_id].get("actions", []))
    return (len(actions) >= 3 and actions[-1] == 0 and
            all(action in (0, 1, 2, 3) for action in actions) and
            0 not in actions[:-1])


def policy_trajectories(paths: list[Path], episodes: dict[str, dict]) -> set[tuple[str, str]]:
    used = set()
    for path in paths:
        steps, groups = [], Counter()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    eid = str(info["episode_id"])
                    if eid not in episodes or info["data_source"] != "r2r":
                        raise ValueError(f"unknown policy episode {eid}")
                    groups[eid] += 1
                    episode = episodes[eid]
                    used.add((str(episode["scene_id"]), str(episode["trajectory_id"])))
        if steps != list(range(1, 129)) or len(groups) != 512 or any(count != 4 for count in groups.values()):
            raise ValueError(f"incomplete group-four policy source {path}")
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--policy-rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.scene_split.read_text())
    if split["schema"] != "stop_history_scene_split_v1":
        raise ValueError("unexpected scene split")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["episodes"]
    if digest(args.train_dataset) != split["train_dataset_sha256"]:
        raise ValueError("train dataset changed since scene freeze")
    with gzip.open(args.ground_truth, "rt", encoding="utf-8") as stream:
        ground_truth = json.load(stream)
    episodes = {str(row["episode_id"]): row for row in rows}
    if len(episodes) != len(rows):
        raise ValueError("duplicate episode ID")
    used_policy = policy_trajectories(args.policy_rollout, episodes)
    by_pose = defaultdict(list)
    by_trajectory = defaultdict(list)
    for row in rows:
        by_pose[pose_key(row)].append(row)
        by_trajectory[(str(row["scene_id"]), str(row["trajectory_id"]))].append(row)
    scene_to_part = {scene: name for name, members in split["split"].items() for scene in members}
    if len(scene_to_part) != 58:
        raise ValueError("scene partition overlap or missing scenes")
    inventory = {name: Counter() for name in TARGET}
    candidates = defaultdict(list)
    for trajectory, alternatives in by_trajectory.items():
        scene = trajectory[0]
        if scene not in scene_to_part:
            continue
        part = scene_to_part[scene]
        inventory[part]["all_trajectories"] += 1
        if part != "fit" and trajectory in used_policy:
            inventory[part]["excluded_policy_trajectory"] += 1
            continue
        eligible = [row for row in alternatives if eligible_gt(row, ground_truth)]
        if not eligible:
            continue
        inventory[part]["eligible_expert_trajectory"] += 1
        primary = min(eligible, key=lambda row: order(str(row["episode_id"])))
        swaps = [other for other in by_pose[pose_key(primary)]
                 if str(other["trajectory_id"]) != trajectory[1] and
                 goal_gap(primary, other) >= 3.5 and
                 str(other["instruction"]["instruction_text"]).strip() !=
                 str(primary["instruction"]["instruction_text"]).strip()]
        if not swaps:
            continue
        inventory[part]["eligible_with_natural_swap"] += 1
        swap = min(swaps, key=lambda row: order(str(row["episode_id"])))
        candidates[part].append({
            "episode_id": str(primary["episode_id"]),
            "trajectory_id": trajectory[1], "scene_id": scene,
            "instruction": str(primary["instruction"]["instruction_text"]),
            "swap_episode_id": str(swap["episode_id"]),
            "wrong_instruction": str(swap["instruction"]["instruction_text"]),
            "goal_gap_m": goal_gap(primary, swap),
            "start_geodesic_distance_m": float(primary["info"]["geodesic_distance"]),
            "motion_actions": len(ground_truth[str(primary["episode_id"])]["actions"]) - 1,
        })
    selected = {}
    for part, target in TARGET.items():
        values = sorted(candidates[part], key=lambda row: order(row["scene_id"] + ":" + row["trajectory_id"]))
        if len(values) < target:
            raise ValueError(f"{part}: only {len(values)} eligible with swap; need {target}; inventory={inventory[part]}")
        selected[part] = values[:target]
    selected_keys = {(row["scene_id"], row["trajectory_id"])
                     for values in selected.values() for row in values}
    if len(selected_keys) != sum(TARGET.values()):
        raise ValueError("trajectory leakage across splits")
    result = {
        "schema": "stop_history_expert_manifest_v1",
        "selection_rule": "one hashed natural instruction per (scene,trajectory), start geodesic >=4m, valid GT actions, natural same-start wrong instruction with goal gap >=3.5m; hash-order fixed targets",
        "source_sha256": {
            "train_dataset": digest(args.train_dataset),
            "ground_truth": digest(args.ground_truth),
            "scene_split": digest(args.scene_split),
            "policy_rollouts": {path.parent.name: digest(path) for path in args.policy_rollout},
        },
        "targets": TARGET, "inventory": {part: dict(counts) for part, counts in inventory.items()},
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"targets": TARGET,
                      "inventory": result["inventory"],
                      "selected": {key: len(value) for key, value in selected.items()}}, indent=2))


if __name__ == "__main__":
    main()
