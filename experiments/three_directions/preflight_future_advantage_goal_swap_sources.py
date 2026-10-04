"""Count start-matched alternative goals without opening images or labels.

This is a necessary-coverage bound for the planned instruction goal-swap
audit. It does not verify route-order reversal and cannot pass that audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def rounded(values: list[float], decimals: int = 4) -> tuple[float, ...]:
    return tuple(round(float(value), decimals) for value in values)


def first_goal(episode: dict) -> tuple[float, ...]:
    goals = episode["goals"]
    if not goals:
        raise ValueError("episode has no goal")
    return rounded(goals[0]["position"])


def matching_ids(episodes: list[dict], *, min_goal_separation: float = 1.) -> dict:
    by_start = defaultdict(list)
    for episode in episodes:
        key = (episode["scene_id"], rounded(episode["start_position"]),
               rounded(episode["start_rotation"]))
        by_start[key].append((str(episode["episode_id"]), first_goal(episode)))
    matched = set()
    pairs = 0
    eligible_starts = 0
    for rows in by_start.values():
        start_eligible = False
        for index, (left_id, left_goal) in enumerate(rows):
            for right_id, right_goal in rows[index + 1:]:
                if math.dist(left_goal, right_goal) >= min_goal_separation:
                    matched.update((left_id, right_id))
                    pairs += 1
                    start_eligible = True
        eligible_starts += int(start_eligible)
    return {"episodes": len(episodes), "start_groups": len(by_start),
            "start_groups_with_alternative_goal": eligible_starts,
            "different_goal_pairs": pairs,
            "episode_ids_with_alternative_goal": len(matched),
            "matching_episode_ids": sorted(matched, key=int)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    split = json.loads(args.scene_split.read_text())["scene_split"]
    prospective = json.loads(args.prospective.read_text())
    if prospective["schema"] != "process_reward_prospective_train_scene_audit_v1":
        raise ValueError("unexpected prospective ID-only manifest")
    selected_ids = set()
    with args.rollout.open() as stream:
        for line in stream:
            selected_ids.update(str(info["episode_id"])
                                for info in json.loads(line)["info"])
    if len(selected_ids) != 512:
        raise ValueError("expected 512 exact training episode IDs")
    parts = dict(split)
    parts["prospective"] = sorted(prospective["scenes"])
    if len({scene for scenes in parts.values() for scene in scenes}) != \
            sum(len(scenes) for scenes in parts.values()):
        raise ValueError("partitions are not scene-disjoint")
    result = {"schema": "future_advantage_goal_swap_source_preflight_v1",
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (("dataset", args.dataset),
                                                   ("scene_split", args.scene_split),
                                                   ("prospective", args.prospective),
                                                   ("rollout", args.rollout))},
              "min_alternative_goal_separation_m": 1.,
              "selected_train_episode_count": len(selected_ids),
              "partitions": {},
              "interpretation": "Necessary start-matched source coverage only; no route reversal or model prediction tested"}
    for part, scenes in parts.items():
        part_episodes = [episode for episode in episodes
                         if episode["scene_id"] in set(scenes)]
        all_count = matching_ids(part_episodes)
        selected_count = matching_ids([
            episode for episode in part_episodes
            if str(episode["episode_id"]) in selected_ids])
        result["partitions"][part] = {
            "scenes": len(scenes),
            "selected_routes_with_any_dataset_alternative_goal": len(
                set(all_count["matching_episode_ids"]) & selected_ids),
            "all_dataset": {key: value for key, value in all_count.items()
                            if key != "matching_episode_ids"},
            "selected_training_rows": {
                key: value for key, value in selected_count.items()
                if key != "matching_episode_ids"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
