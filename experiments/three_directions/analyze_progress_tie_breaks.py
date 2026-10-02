"""Offline training-return diagnostic for the proposed terminal progress bonus.

Episode metadata supplies initial geodesic distance as a proxy for Habitat's
reset-time metric. This re-scores saved rollouts; it does not train or evaluate
a progress-reward policy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from progress_reward import bounded_terminal_progress


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_rollouts(path: Path, initial_distance: dict[str, float]) -> tuple[dict, list[set[str]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 128
    assert [row["step"] for row in rows] == list(range(1, 129))
    counts = {
        "groups": 0,
        "original_equal_return_groups": 0,
        "equal_return_groups_broken": 0,
        "groups_with_shaped_return_variance": 0,
        "original_preference_flips": 0,
        "invalid_distance_rollouts": 0,
    }
    broken_tie_gaps = []
    per_step_ids = []
    seen_ids = set()
    for row in rows:
        by_id = defaultdict(list)
        for item in row["info"]:
            assert "geodesic_progress_reward" not in item["reward_components"]
            by_id[str(item["episode_id"])].append(item)
        assert len(by_id) == 4 and all(len(pair) == 2 for pair in by_id.values())
        per_step_ids.append(set(by_id))
        assert not seen_ids.intersection(by_id)
        seen_ids.update(by_id)
        for episode_id, pair in by_id.items():
            original = [float(item["total_reward"]) for item in pair]
            shaped = []
            for index, item in enumerate(pair):
                bonus, invalid = bounded_terminal_progress(
                    initial_distance[episode_id],
                    float(item["distance_to_goal"]), 1.0)
                counts["invalid_distance_rollouts"] += invalid
                shaped.append(original[index] + bonus)
            original_delta = original[0] - original[1]
            shaped_delta = shaped[0] - shaped[1]
            tied = abs(original_delta) < 1e-8
            counts["groups"] += 1
            counts["original_equal_return_groups"] += tied
            counts["equal_return_groups_broken"] += tied and abs(shaped_delta) > 1e-8
            if tied and abs(shaped_delta) > 1e-8:
                broken_tie_gaps.append(abs(shaped_delta))
            counts["groups_with_shaped_return_variance"] += abs(shaped_delta) > 1e-8
            counts["original_preference_flips"] += not tied and original_delta * shaped_delta < 0
    assert counts["groups"] == len(seen_ids) == 512
    counts["broken_tie_bonus_gap_median_reward_units"] = statistics.median(broken_tie_gaps)
    for threshold in (0.025, 0.05, 0.1):
        counts[f"broken_ties_gap_gt_{threshold:g}"] = sum(
            gap > threshold for gap in broken_tie_gaps)
    return counts, per_step_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.episodes, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    initial_distance = {str(episode["episode_id"]):
                        float(episode["info"]["geodesic_distance"])
                        for episode in episodes}
    assert len(initial_distance) == len(episodes) == 10819
    output = {
        "seed": args.seed,
        "interpretation": "Offline re-scoring of training rollouts only; no held-out policy result.",
        "initial_distance_source": "R2R train episode metadata, proxy for runtime reset distance",
        "episode_metadata_sha256": file_hash(args.episodes),
        "progress_weight": 1.0,
        "arms": {},
    }
    all_step_ids = []
    for arm in ("branch", "branch_control"):
        name = f"three_directions_{arm}_128step"
        if args.seed != 11:
            name += f"_seed{args.seed}"
        run = args.root / "runlogs" / name
        assert (run / "completed").exists() and not (run / "failed").exists()
        rollout = args.root / "verl_checkpoints" / name / "rollout.jsonl"
        counts, step_ids = analyze_rollouts(rollout, initial_distance)
        output["arms"][arm] = {"rollout_sha256": file_hash(rollout), **counts}
        all_step_ids.append(step_ids)
    assert all_step_ids[0] == all_step_ids[1]
    output["matched_episode_sets_at_each_step"] = 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
