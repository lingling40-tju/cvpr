"""Freeze scene-disjoint train-only splits before fitting a STOP/progress head."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


SALT = "stop-history-v1:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.preflight.read_bytes()
    preflight = json.loads(source_bytes)
    if preflight["schema"] != "train_only_stop_supervision_preflight_v1":
        raise ValueError("wrong preflight schema")
    scenes = sorted(preflight["scene_counts"],
                    key=lambda scene: hashlib.sha256((SALT + scene).encode()).hexdigest())
    if len(scenes) != 58:
        raise ValueError("unexpected number of policy-train scenes")
    split = {"development": scenes[:9], "audit": scenes[9:18], "fit": scenes[18:]}
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        train = json.load(stream)["episodes"]
    train_counts = Counter(str(row["scene_id"]) for row in train)
    train_trajectories = {str(row["scene_id"]): set() for row in train}
    for row in train:
        train_trajectories[str(row["scene_id"])].add(str(row["trajectory_id"]))
    split_counts = {}
    for name, members in split.items():
        split_counts[name] = {
            "scenes": len(members),
            "train_episode_ids": sum(train_counts[scene] for scene in members),
            "train_unique_trajectories": sum(len(train_trajectories[scene]) for scene in members),
            "policy_positive_rollouts": sum(preflight["scene_counts"][scene].get("positive_success_stop", 0)
                                            for scene in members),
            "policy_negative_rollouts": sum(preflight["scene_counts"][scene].get("negative_failed_stop", 0)
                                            for scene in members),
            "policy_positive_unique_episode_ids": sum(
                preflight["scene_unique_episodes_by_label"][scene].get("positive_success_stop", 0)
                for scene in members),
            "policy_negative_unique_episode_ids": sum(
                preflight["scene_unique_episodes_by_label"][scene].get("negative_failed_stop", 0)
                for scene in members),
        }
    result = {
        "schema": "stop_history_scene_split_v1",
        "preflight_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "train_dataset_sha256": preflight["train_dataset_sha256"],
        "scene_order_rule": "ascending SHA-256 of stop-history-v1: + scene_id; first 9 development, next 9 audit, remaining 40 fit",
        "split": split,
        "counts": split_counts,
        "interpretation": "Counts from three policy seeds reuse episode IDs and are not independent samples. The audit split has fewer than 100 unique rollout-positive and rollout-negative episode IDs; collect additional train-scene expert/policy views before testing the predeclared STOP gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(split_counts, indent=2))


if __name__ == "__main__":
    main()
