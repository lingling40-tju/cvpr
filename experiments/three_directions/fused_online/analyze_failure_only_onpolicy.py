"""Diagnose failure-only reward order on group-four training rollouts.

Simulator distance is read only as an analysis label and never enters the
reward server. This is not a held-out navigation evaluation.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def read_rows(path: Path, incomplete_tail: bool) -> list[dict]:
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if incomplete_tail and index == len(lines) - 1:
                break
            raise
    return result


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--allow-incomplete-tail", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = read_rows(args.rollout, args.allow_incomplete_tail)
    if args.expected_steps is not None and len(records) != args.expected_steps:
        raise ValueError("training step coverage mismatch")
    ids = set()
    groups = successes = failures = all_failed = mixed = 0
    priority_violations = 0
    bonus = []
    paired = {"all_failed_pairs": collections.Counter(),
              "selected_large_gap_pairs": collections.Counter()}
    for step, record in enumerate(records, 1):
        if record["step"] != step:
            raise ValueError("nonsequential training step")
        by_episode = collections.defaultdict(list)
        for item in record["info"]:
            by_episode[str(item["episode_id"])].append(item)
        if len(by_episode) != 4 or ids & set(by_episode):
            raise ValueError("unexpected episode grouping or reuse")
        ids.update(by_episode)
        for items in by_episode.values():
            if len(items) != 4:
                raise ValueError("group size is not four")
            groups += 1
            positive = [item for item in items if item["task_success"]]
            negative = [item for item in items if not item["task_success"]]
            successes += len(positive)
            failures += len(negative)
            all_failed += not positive
            mixed += bool(positive and negative)
            for item in positive:
                if item["fused_reward"]["status"] != "disabled" or \
                        float(item["reward_components"]["fused_bonus"]) != 0:
                    raise ValueError("success received failure-only bonus")
            for item in negative:
                score = item["fused_reward"]
                value = float(score["bonus"])
                if score["status"] != "ok" or not 0 < value < 1 or \
                        not math.isfinite(float(item["distance_to_goal"])):
                    raise ValueError("invalid failure score or distance")
                bonus.append(value)
            for winner in positive:
                for loser in negative:
                    priority_violations += (float(winner["total_reward"]) <=
                                            float(loser["total_reward"]))
            for i, first in enumerate(negative):
                for second in negative[i + 1:]:
                    d1, d2 = (float(item["distance_to_goal"])
                              for item in (first, second))
                    if abs(d1 - d2) < .05:
                        continue
                    near, far = (first, second) if d1 < d2 else (second, first)
                    delta = (float(near["fused_reward"]["bonus"]) -
                             float(far["fused_reward"]["bonus"]))
                    value = 1 if delta > 0 else -1 if delta < 0 else 0
                    if not positive:
                        paired["all_failed_pairs"][value] += 1
                    if min(d1, d2) >= 3.5 and abs(d1 - d2) >= 1.5:
                        paired["selected_large_gap_pairs"][value] += 1
    ranking = {}
    for name, count in paired.items():
        wins, ties, losses = count[1], count[0], count[-1]
        ranking[name] = {"nearer_above_farther": wins, "ties": ties,
                         "farther_above_nearer": losses,
                         "pairs": wins + ties + losses,
                         "win_rate_excluding_ties": fraction(wins, wins + losses)}
    report = {"schema": "failure_only_group4_training_signal_v1",
              "interpretation": "Training-only analysis; simulator distance is a diagnostic label, not a reward input. No val-unseen navigation claim.",
              "steps": len(records), "group_size": 4,
              "episode_groups": groups,
              "successful_rollouts": successes,
              "unsuccessful_rollouts": failures,
              "all_failure_groups": all_failed,
              "mixed_outcome_groups": mixed,
              "success_reward_priority_violations_in_mixed_groups": priority_violations,
              "failure_bonus_mean": statistics.mean(bonus) if bonus else None,
              "failure_bonus_sd": statistics.stdev(bonus) if len(bonus) > 1 else None,
              "ranking": ranking}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
