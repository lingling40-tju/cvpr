"""Quantify final-distance differences hidden by equal training returns."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def one_arm(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 128
    assert [row["step"] for row in rows] == list(range(1, 129))
    counts = {
        "groups": 0,
        "equal_return_groups": 0,
        "equal_return_final_distance_gap_gt_0_25m": 0,
        "equal_return_final_distance_gap_gt_0_5m": 0,
        "equal_return_final_distance_gap_gt_1m": 0,
        "equal_return_both_failed_gap_gt_0_5m": 0,
        "nonfinite_final_distance_groups": 0,
    }
    per_step_ids = []
    for row in rows:
        by_id = defaultdict(list)
        for item in row["info"]:
            by_id[str(item["episode_id"])].append(item)
        assert len(by_id) == 4 and all(len(group) == 2 for group in by_id.values())
        per_step_ids.append(set(by_id))
        for pair in by_id.values():
            counts["groups"] += 1
            distances = [float(item["distance_to_goal"]) for item in pair]
            if not all(math.isfinite(distance) for distance in distances):
                counts["nonfinite_final_distance_groups"] += 1
                continue
            equal_return = abs(float(pair[0]["total_reward"]) - float(pair[1]["total_reward"])) < 1e-8
            if not equal_return:
                continue
            counts["equal_return_groups"] += 1
            gap = abs(distances[0] - distances[1])
            counts["equal_return_final_distance_gap_gt_0_25m"] += gap > 0.25
            counts["equal_return_final_distance_gap_gt_0_5m"] += gap > 0.5
            counts["equal_return_final_distance_gap_gt_1m"] += gap > 1.0
            counts["equal_return_both_failed_gap_gt_0_5m"] += (
                gap > 0.5 and not bool(pair[0]["task_success"])
                and not bool(pair[1]["task_success"])
            )
    assert counts["groups"] == 512
    return {"counts": counts, "per_step_ids": per_step_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.seed in (11, 22, 33)
    result = {
        "seed": args.seed,
        "steps": 128,
        "interpretation": "Training-rollout diagnosis only; not a held-out navigation score.",
        "arms": {},
    }
    step_ids = []
    for arm in ("branch", "branch_control"):
        name = f"three_directions_{arm}_128step"
        if args.seed != 11:
            name += f"_seed{args.seed}"
        run = args.root / "runlogs" / name
        assert (run / "completed").exists() and not (run / "failed").exists()
        observed = one_arm(args.root / "verl_checkpoints" / name / "rollout.jsonl")
        result["arms"][arm] = observed["counts"]
        step_ids.append(observed["per_step_ids"])
    assert step_ids[0] == step_ids[1]
    result["matched_episode_sets_at_each_step"] = 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
