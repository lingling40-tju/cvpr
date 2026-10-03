"""Freeze a same-scene, different-start wrong goal before teacher scoring.

This tests whether the exact-same-start route matcher transfers to a
counterfactual available for every group-four training instruction. Source
instructions and RGB remain on the experiment machine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from prepare_qwen3_policy_route_manifest import digest, pose


def pick(original: dict, by_scene: dict[str, list[dict]]) -> tuple[str, float, float]:
    eid = str(original["episode_id"])
    choices = []
    for other in by_scene[str(original["scene_id"])]:
        wrong_id = str(other["episode_id"])
        if wrong_id == eid or pose(other) == pose(original) or \
                other["instruction"]["instruction_text"].strip() == \
                original["instruction"]["instruction_text"].strip():
            continue
        start_gap = math.dist(original["start_position"], other["start_position"])
        goal_gap = math.dist(original["goals"][0]["position"],
                             other["goals"][0]["position"])
        if start_gap < 1 or goal_gap < 4:
            continue
        tie = hashlib.sha256(
            f"qwen3-scene-fallback-v1:{eid}:{wrong_id}".encode()).hexdigest()
        choices.append((start_gap, tie, wrong_id, goal_gap))
    if not choices:
        raise ValueError(f"no same-scene fallback for episode {eid}")
    start_gap, _, wrong_id, goal_gap = min(choices)
    return wrong_id, start_gap, goal_gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.policy_manifest.read_text())
    if source["schema"] != "qwen3_policy_route_manifest_v1" or \
            source["source_sha256"]["train_dataset"] != digest(args.train_dataset):
        raise ValueError("source manifest/dataset mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate train episode ID")
    by_scene = defaultdict(list)
    for row in episodes:
        by_scene[str(row["scene_id"])].append(row)
    output = json.loads(args.policy_manifest.read_text())
    output["selection"] = (
        "exploratory same-scene fallback: different start >=1m, different "
        "goal >=4m; nearest eligible start then SHA tie-break; all route "
        "records, labels, and split memberships identical to parent manifest")
    output["parent_policy_manifest_sha256"] = digest(args.policy_manifest)
    summary = {}
    for part, groups in output["selected"].items():
        gaps = []
        for group in groups:
            original = by_id[group["episode_id"]]
            wrong_id, start_gap, goal_gap = pick(original, by_scene)
            group["wrong_episode_id"] = wrong_id
            group["goal_gap_m_for_selection_only"] = goal_gap
            group["wrong_start_gap_m_for_selection_only"] = start_gap
            gaps.append(start_gap)
        summary[part] = {"groups": len(groups),
                         "min_wrong_start_gap_m": min(gaps),
                         "max_wrong_start_gap_m": max(gaps)}
    output["scene_fallback_inventory"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"inventory": summary, "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
