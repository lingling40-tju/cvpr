"""Freeze natural wrong-goal instructions for prefix development groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    if source["schema"] != "policy_group_relative_manifest_v1" or \
            source["train_dataset_sha256"] != digest(args.train_dataset):
        raise ValueError("train source mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("ambiguous train episode ID")
    by_scene = defaultdict(list)
    for row in episodes:
        by_scene[str(row["scene_id"])].append(row)
    source_ids = sorted({str(row["episode_id"]) for row in
                         source["selected"]["development"]})
    swaps = {}
    for eid in source_ids:
        original = by_id[eid]
        scene = str(original["scene_id"])
        goal = original["goals"][0]["position"]
        start = original["start_position"]
        original_instruction = original["instruction"]["instruction_text"].strip()
        choices = []
        for other in by_scene[scene]:
            if str(other["episode_id"]) == eid:
                continue
            wrong_instruction = other["instruction"]["instruction_text"].strip()
            goal_gap = math.dist(goal, other["goals"][0]["position"])
            if wrong_instruction == original_instruction or goal_gap < 4:
                continue
            start_gap = math.dist(start, other["start_position"])
            rank = hashlib.sha256(
                f"group4-prefix-development-wrong-goal-v1:{eid}:{other['episode_id']}".encode()).hexdigest()
            choices.append((rank, other, goal_gap, start_gap))
        if not choices:
            raise ValueError(f"no wrong-goal natural instruction {eid}")
        near_start = [item for item in choices if item[3] <= .5]
        _, chosen, goal_gap, start_gap = min(near_start or choices)
        swaps[eid] = {"scene_id": scene,
                      "original_instruction": original_instruction,
                      "swapped_episode_id": str(chosen["episode_id"]),
                      "swapped_instruction": chosen["instruction"]["instruction_text"].strip(),
                      "goal_separation_m_for_selection_only": goal_gap,
                      "start_separation_m_for_analysis_only": start_gap,
                      "same_start_within_0_5m_available": any(
                          item[3] <= .5 for item in choices)}
    result = {"schema": "group4_prefix_development_instruction_swaps_v1",
              "source_manifest_sha256": digest(args.manifest),
              "train_dataset_sha256": digest(args.train_dataset),
              "selection": "prefer natural same-start (within 0.5 m) different-goal instruction when available, otherwise same-scene; SHA rank within tier; goal gap at least 4 m",
              "episode_count": len(swaps),
              "same_start_within_0_5m_available": sum(
                  row["same_start_within_0_5m_available"] for row in swaps.values()),
              "swaps": swaps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episode_count": len(swaps),
                      "same_start_within_0_5m_available":
                          result["same_start_within_0_5m_available"],
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
