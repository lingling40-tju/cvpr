"""Audit train-only endpoint and wrong-instruction labels before GPU fitting.

The GT final location is used only as a conservative lower bound on
distance to the swapped goal: Euclidean >3.5 m guarantees geodesic >3.5 m.
This script does not open images or use val-unseen data.
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "stop_history_expert_manifest_v1" or \
            digest(args.train_dataset) != manifest["source_sha256"]["train_dataset"] or \
            digest(args.ground_truth) != manifest["source_sha256"]["ground_truth"]:
        raise ValueError("source checksum mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row for row in json.load(stream)["episodes"]}
    with gzip.open(args.ground_truth, "rt", encoding="utf-8") as stream:
        ground_truth = json.load(stream)
    manifest_sha = digest(args.manifest)
    counts = {}
    labels = {}
    for part, plans in manifest["selected"].items():
        part_counts = Counter()
        part_rows = []
        for plan in plans:
            eid = str(plan["episode_id"])
            record = json.loads((args.records_root / part / "records" / f"{eid}.json").read_text())
            if record["manifest_sha256"] != manifest_sha or record["episode_id"] != eid:
                raise ValueError(f"record checksum/ID mismatch {part}/{eid}")
            if not record["turns"] or record["turn_count"] != len(record["turns"]):
                raise ValueError(f"empty or invalid history {part}/{eid}")
            correct = episodes[eid]
            swapped = episodes[str(plan["swap_episode_id"])]
            final_position = ground_truth[eid]["locations"][-1]
            wrong_goal = swapped["goals"][0]["position"]
            correct_goal = correct["goals"][0]["position"]
            wrong_euclidean = math.dist(final_position, wrong_goal)
            correct_euclidean = math.dist(final_position, correct_goal)
            if correct_euclidean > record["end_distance_to_goal_for_label_only"] + 0.25:
                raise ValueError(f"GT final position disagrees with Habitat metric {part}/{eid}")
            within_budget = record["turn_count"] <= 12
            valid_stop = record["end_within_3m"] and \
                record["start_distance_to_goal_for_label_only"] >= 3.5
            safe_swap = wrong_euclidean >= 3.5
            part_counts["trajectories"] += 1
            if within_budget:
                part_counts["within_12_turns"] += 1
                if valid_stop:
                    part_counts["positive_stop_and_far_start"] += 1
                if valid_stop and safe_swap:
                    part_counts["safe_natural_wrong_instruction"] += 1
            part_rows.append({
                "episode_id": eid, "scene_id": record["scene_id"],
                "trajectory_id": record["trajectory_id"],
                "within_12_turns": within_budget,
                "stop_labels_valid": valid_stop,
                "safe_wrong_instruction": safe_swap,
                "wrong_goal_euclidean_lower_bound_m": wrong_euclidean,
                "start_distance_to_goal_m_for_label_only":
                    record["start_distance_to_goal_for_label_only"],
                "mid_distance_to_goal_m_for_label_only":
                    record["turns"][max(0, record["turn_count"] // 2 - 1)]["distance_to_goal_for_label_only"],
                "end_distance_to_goal_m_for_label_only":
                    record["end_distance_to_goal_for_label_only"],
            })
        counts[part] = dict(part_counts)
        labels[part] = part_rows
    result = {
        "schema": "stop_history_train_only_label_audit_v1",
        "manifest_sha256": manifest_sha,
        "selection": "Only <=12-turn trajectories enter the first representation screen; swapped instruction is evaluated only when GT final location is at least 3.5m Euclidean from its goal.",
        "counts": counts,
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
