"""Freeze start-matched, different-goal train-audit IDs before model scores.

This manifest contains episode identities only. It does not select on policy
outcomes, geodesic route reversal, reward-model predictions, or RGB.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from preflight_future_advantage_goal_swap_sources import first_goal, rounded


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pairs(episodes: list[dict], audit_scenes: set[str]) -> list[dict]:
    by_start = defaultdict(list)
    for episode in episodes:
        if episode["scene_id"] not in audit_scenes:
            continue
        key = (episode["scene_id"], rounded(episode["start_position"]),
               rounded(episode["start_rotation"]))
        by_start[key].append(episode)
    result = []
    for (scene, _, _), rows in sorted(by_start.items()):
        candidates = []
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                distance = math.dist(first_goal(left), first_goal(right))
                if distance >= 1.:
                    left_id, right_id = sorted((int(left["episode_id"]),
                                                int(right["episode_id"])))
                    candidates.append((-distance, left_id, right_id))
        if not candidates:
            continue
        negative_distance, left_id, right_id = min(candidates)
        result.append({"scene_id": scene, "episode_ids": [left_id, right_id],
                       "goal_separation_m": round(-negative_distance, 6)})
    result.sort(key=lambda row: (row["scene_id"], row["episode_ids"]))
    for index, row in enumerate(result, 1):
        row["pair_id"] = f"goal_swap_{index:03d}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset", "scene-split", "prospective", "source-preflight",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.source_preflight.read_text())
    if preflight.get("schema") != "future_advantage_goal_swap_source_preflight_v1" or \
            preflight["source_sha256"]["dataset"] != digest(args.dataset) or \
            preflight["source_sha256"]["scene_split"] != digest(args.scene_split) or \
            preflight["source_sha256"]["prospective"] != digest(args.prospective):
        raise ValueError("goal-swap source preflight changed")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    split = json.loads(args.scene_split.read_text())["scene_split"]
    prospective = json.loads(args.prospective.read_text())
    audit_scenes = set(split["audit"])
    if audit_scenes & set(split["fit"]) or \
            audit_scenes & set(split["development"]) or \
            audit_scenes & set(prospective["scenes"]):
        raise ValueError("goal-swap audit scenes overlap another partition")
    pairs = select_pairs(episodes, audit_scenes)
    expected = preflight["partitions"]["audit"]["all_dataset"][
        "start_groups_with_alternative_goal"]
    ids = [eid for row in pairs for eid in row["episode_ids"]]
    if len(pairs) != expected or len(ids) != len(set(ids)) or \
            any(row["goal_separation_m"] < 1. for row in pairs):
        raise ValueError("goal-swap source pair coverage changed")
    manifest = {
        "schema": "future_advantage_goal_swap_id_manifest_v1",
        "source_sha256": {"dataset": digest(args.dataset),
                          "scene_split": digest(args.scene_split),
                          "prospective": digest(args.prospective),
                          "source_preflight": digest(args.source_preflight)},
        "audit_scenes": len(audit_scenes), "start_matched_pairs": len(pairs),
        "unique_episode_ids": len(ids),
        "planned_rollouts_per_episode": 4,
        "planned_sft_rollouts": len(ids) * 4,
        "pairs": pairs,
        "interpretation": "ID-only candidate sources; collect routes only after the learned development gate passes; route reversal and model grounding remain untested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "audit_scenes", "start_matched_pairs", "unique_episode_ids",
        "planned_sft_rollouts")}, indent=2))


if __name__ == "__main__":
    main()
