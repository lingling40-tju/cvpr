"""Train-scene arithmetic for a small stop preference on frozen progress.

Unlike a hard signed timeout rule, the fixed 0.1 offset leaves the progress
score active for both unsuccessful STOP and timeout trajectories. The probe
uses completed group-four training rollouts only; no retraining, audit, or
val-unseen label is involved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


STOP = "stopped but goal not reached."
TIMEOUT = "number of turns exceeded."
FORMAT = "unexpected format."
EPSILON = 0.1


def reward(item: dict, rule: str) -> float:
    total = float(item["total_reward"])
    if item["task_success"]:
        return total
    b = float(item["fused_reward"]["bonus"])
    old = float(item["reward_components"]["fused_bonus"])
    if not math.isfinite(b) or not 0 <= b <= 1 or abs(old - b) > 1e-6:
        raise ValueError("invalid frozen bonus")
    if rule == "raw_failure":
        bonus = b
    elif rule == "soft_stop_prior":
        reason = item["end_reason"]
        bonus = (b + EPSILON if reason == STOP else
                 b - EPSILON if reason == TIMEOUT else
                 -1 if reason == FORMAT else None)
        if bonus is None:
            raise ValueError(f"unknown termination reason: {reason}")
    else:
        raise ValueError("unknown reward rule")
    return total - old + bonus


def summarize(groups: list[list[dict]], rule: str) -> dict:
    group_count = Counter()
    large = Counter()
    cross = Counter()
    for items in groups:
        values = [reward(item, rule) for item in items]
        success = [i for i, item in enumerate(items) if item["task_success"]]
        failure = [i for i, item in enumerate(items) if not item["task_success"]]
        group_count["groups"] += 1
        group_count["all_failure_groups"] += not success
        group_count["all_failure_groups_with_variance"] += (
            not success and len({round(x, 7) for x in values}) > 1)
        group_count["success_priority_violations"] += sum(
            values[i] <= values[j] for i in success for j in failure)
        if success:
            continue
        for position, i in enumerate(failure):
            for j in failure[position + 1:]:
                a, b = items[i], items[j]
                if {a["end_reason"], b["end_reason"]} == {STOP, TIMEOUT}:
                    stop, timeout = (i, j) if a["end_reason"] == STOP else (j, i)
                    cross["pairs"] += 1
                    cross["stop_above_timeout"] += values[stop] > values[timeout]
                    cross["ties"] += values[stop] == values[timeout]
                d1, d2 = float(a["distance_to_goal"]), float(b["distance_to_goal"])
                if not all(math.isfinite(x) for x in (d1, d2)) or \
                        min(d1, d2) < 3.5 or abs(d1 - d2) < 1.5:
                    continue
                near, far = (i, j) if d1 < d2 else (j, i)
                sign = (1 if values[near] > values[far] else
                        -1 if values[near] < values[far] else 0)
                large[sign] += 1
    wins, ties, losses = (large[sign] for sign in (1, 0, -1))
    return {**dict(group_count),
            "large_gap_nearer_rank": {
                "nearer_above_farther": wins, "ties": ties,
                "farther_above_nearer": losses, "pairs": wins + ties + losses,
                "win_rate_excluding_ties": wins / (wins + losses)
                if wins + losses else None},
            "stop_vs_timeout_in_all_failed_groups": {
                **dict(cross),
                "stop_win_rate_excluding_ties":
                    cross["stop_above_timeout"] / (cross["pairs"] - cross["ties"])
                    if cross["pairs"] > cross["ties"] else None}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollout.read_text().splitlines()]
    if len(rows) != 64 or [row["step"] for row in rows] != list(range(1, 65)):
        raise ValueError("incomplete prior group-four training")
    groups = []
    seen = set()
    for row in rows:
        by_episode = defaultdict(list)
        for item in row["info"]:
            by_episode[str(item["episode_id"])].append(item)
        if len(by_episode) != 4 or seen & set(by_episode) or \
                {len(items) for items in by_episode.values()} != {4}:
            raise ValueError("group-four coverage mismatch")
        seen.update(by_episode)
        groups.extend(by_episode.values())
    if len(groups) != 256:
        raise ValueError("incomplete episode coverage")
    table = {rule: summarize(groups, rule)
             for rule in ("raw_failure", "soft_stop_prior")}
    original, candidate = (table[rule] for rule in ("raw_failure", "soft_stop_prior"))
    old_rank = original["large_gap_nearer_rank"]["win_rate_excluding_ties"]
    new_rank = candidate["large_gap_nearer_rank"]["win_rate_excluding_ties"]
    old_stop = original["stop_vs_timeout_in_all_failed_groups"]["stop_win_rate_excluding_ties"]
    new_stop = candidate["stop_vs_timeout_in_all_failed_groups"]["stop_win_rate_excluding_ties"]
    gate = {
        "all_failure_variance_at_least_95pct":
            candidate["all_failure_groups_with_variance"] >=
            .95 * candidate["all_failure_groups"],
        "large_gap_ranking_drop_at_most_2pp":
            new_rank is not None and old_rank is not None and
            new_rank >= old_rank - .02 - 1e-8,
        "stop_preference_gain_at_least_5pp":
            new_stop is not None and old_stop is not None and
            new_stop >= old_stop + .05 - 1e-8,
        "success_priority_preserved":
            candidate["success_priority_violations"] == 0}
    result = {"schema": "soft_stop_prior_train_counterfactual_v1",
              "interpretation": "Train-scene counterfactual arithmetic on a policy trained with the original failure bonus. Group-relative preference can still favor a timeout despite its small negative offset. No new policy or held-out navigation outcome is implied.",
              "steps": 64, "group_size": 4, "episode_groups": len(groups),
              "epsilon": EPSILON,
              "formulas": {"raw_failure": "raw frozen bonus on all failures",
                           "soft_stop_prior": "raw+0.1 on unsuccessful STOP, raw-0.1 on timeout, -1 on format failure"},
              "table": table, "train_only_gate": gate,
              "eligible_for_later_online_screen": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
