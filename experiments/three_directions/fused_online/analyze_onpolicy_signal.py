"""Inspect frozen reward rankings on training rollouts only.

The simulator distance is used here as an analysis label; it is never sent to
the reward service. These diagnostics cannot establish val-unseen performance.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


SCORES = ("bonus", "temporal_score", "visual_score")


def rows(path: Path, allow_incomplete_tail: bool) -> list[dict]:
    output = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_incomplete_tail and index == len(lines) - 1:
                break
            raise
    return output


def compare(higher: dict, lower: dict, key: str) -> int:
    delta = float(higher["fused_reward"][key]) - float(lower["fused_reward"][key])
    if not math.isfinite(delta):
        raise ValueError("nonfinite score")
    return (delta > 0) - (delta < 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--allow-incomplete-tail", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = rows(args.rollout, args.allow_incomplete_tail)
    if args.expected_steps is not None and len(records) != args.expected_steps:
        raise ValueError("training step count mismatch")
    tallies = {name: {score: collections.Counter() for score in SCORES}
               for name in ("success_over_failure", "nearer_failure_over_farther_failure")}
    successes = []
    failures = []
    bonus_std = []
    groups = mixed = all_failures = 0
    ids = set()
    for step, record in enumerate(records, 1):
        if record["step"] != step:
            raise ValueError("nonsequential rollout rows")
        by_episode = collections.defaultdict(list)
        for item in record["info"]:
            by_episode[str(item["episode_id"])].append(item)
        if len(by_episode) != 4 or set(by_episode) & ids:
            raise ValueError("training episode grouping or uniqueness mismatch")
        ids.update(by_episode)
        for items in by_episode.values():
            if len(items) != 4:
                raise ValueError("expected four rollouts per episode")
            groups += 1
            values = [float(item["fused_reward"]["bonus"]) for item in items]
            if not all(math.isfinite(value) and 0 < value < 1 for value in values):
                raise ValueError("invalid bonus")
            bonus_std.append(statistics.pstdev(values))
            winners = [item for item in items if item["task_success"]]
            losers = [item for item in items if not item["task_success"]]
            successes.extend(float(item["fused_reward"]["bonus"]) for item in winners)
            failures.extend(float(item["fused_reward"]["bonus"]) for item in losers)
            mixed += bool(winners and losers)
            all_failures += not winners
            for winner in winners:
                for loser in losers:
                    for score in SCORES:
                        tallies["success_over_failure"][score][
                            compare(winner, loser, score)] += 1
            for index, first in enumerate(losers):
                for second in losers[index + 1:]:
                    first_distance = float(first["distance_to_goal"])
                    second_distance = float(second["distance_to_goal"])
                    if not math.isfinite(first_distance + second_distance):
                        raise ValueError("nonfinite analysis label")
                    if abs(first_distance - second_distance) < 0.05:
                        continue
                    nearer, farther = ((first, second) if first_distance < second_distance
                                       else (second, first))
                    for score in SCORES:
                        tallies["nearer_failure_over_farther_failure"][score][
                            compare(nearer, farther, score)] += 1
    pairs = {}
    for label, by_score in tallies.items():
        pairs[label] = {}
        for score, count in by_score.items():
            wins, ties, losses = count[1], count[0], count[-1]
            pairs[label][score] = {"wins": wins, "ties": ties,
                                   "losses": losses, "pairs": wins + ties + losses,
                                   "win_rate_excluding_ties": wins / (wins + losses)
                                   if wins + losses else None}
    result = {"schema": "fused_training_signal_diagnostic_v1",
              "interpretation": "Training rollouts only. Distance is an analysis label, not part of the reward. No val-unseen or navigation-gain claim.",
              "steps": len(records), "group_size": 4, "episode_groups": groups,
              "rollouts": 4 * groups, "successful_rollouts": len(successes),
              "mixed_outcome_groups": mixed, "all_failure_groups": all_failures,
              "mean_bonus_success": statistics.mean(successes) if successes else None,
              "mean_bonus_failure": statistics.mean(failures) if failures else None,
              "mean_within_group_bonus_std": statistics.mean(bonus_std) if bonus_std else None,
              "pair_rankings": pairs}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
