"""Counterfactual reward arithmetic on completed train-scene rollouts.

No policy is retrained and no val-unseen episode is read. All three reward
rules are applied to the same stored group-four trajectories. Simulator
distance is used only as a diagnostic label, never as proposed reward input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


STOP_REASON = "stopped but goal not reached."
TIMEOUT_REASON = "number of turns exceeded."
FORMAT_REASON = "unexpected format."


def transformed(item: dict, rule: str) -> float:
    if item["task_success"]:
        return float(item["total_reward"])
    raw = float(item["fused_reward"]["bonus"])
    old_bonus = float(item["reward_components"]["fused_bonus"])
    if not math.isfinite(raw) or not 0 <= raw <= 1 or \
            abs(raw - old_bonus) > 1e-6:
        raise ValueError("invalid frozen failure bonus")
    reason = item["end_reason"]
    if rule == "raw_failure":
        bonus = raw
    elif rule == "stop_only":
        bonus = max(0.0, 2 * raw - 1) if reason == STOP_REASON else 0.0
    elif rule == "signed_timeout":
        bonus = (max(0.0, 2 * raw - 1) if reason == STOP_REASON else
                 raw - 1 if reason == TIMEOUT_REASON else
                 -1.0 if reason == FORMAT_REASON else None)
    else:
        raise ValueError("unknown reward rule")
    if bonus is None or not -1 <= bonus <= 1:
        raise ValueError(f"invalid end reason or proposed bonus: {reason}")
    return float(item["total_reward"]) - old_bonus + bonus


def analyze(groups: list[list[dict]], rule: str) -> dict:
    count = Counter()
    pair = Counter()
    for items in groups:
        returns = [transformed(item, rule) for item in items]
        success = [i for i, item in enumerate(items) if item["task_success"]]
        failure = [i for i, item in enumerate(items) if not item["task_success"]]
        count["groups"] += 1
        count["all_failure_groups"] += not success
        count["all_failure_groups_with_return_variance"] += (
            not success and len({round(value, 7) for value in returns}) > 1)
        count["success_priority_violations"] += sum(
            returns[i] <= returns[j] for i in success for j in failure)
        if success:
            continue
        for position, i in enumerate(failure):
            for j in failure[position + 1:]:
                d1 = float(items[i]["distance_to_goal"])
                d2 = float(items[j]["distance_to_goal"])
                if not math.isfinite(d1) or not math.isfinite(d2) or \
                        abs(d1 - d2) < .05:
                    continue
                near, far = (i, j) if d1 < d2 else (j, i)
                sign = (1 if returns[near] > returns[far] else
                        -1 if returns[near] < returns[far] else 0)
                pair[("all_failed_pairs", sign)] += 1
                if min(d1, d2) >= 3.5 and abs(d1 - d2) >= 1.5:
                    pair[("large_gap_pairs", sign)] += 1
    ranking = {}
    for label in ("all_failed_pairs", "large_gap_pairs"):
        wins, ties, losses = (pair[(label, sign)] for sign in (1, 0, -1))
        ranking[label] = {
            "nearer_above_farther": wins, "ties": ties,
            "farther_above_nearer": losses, "pairs": wins + ties + losses,
            "win_rate_excluding_ties": wins / (wins + losses)
            if wins + losses else None}
    return {**dict(count), "ranking": ranking}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollout.read_text().splitlines()]
    if len(rows) != 64 or [row["step"] for row in rows] != list(range(1, 65)):
        raise ValueError("incomplete failure-only training rollout")
    groups = []
    seen = set()
    for row in rows:
        by_episode = defaultdict(list)
        for item in row["info"]:
            by_episode[str(item["episode_id"])].append(item)
        if len(by_episode) != 4 or seen & set(by_episode) or \
                {len(value) for value in by_episode.values()} != {4}:
            raise ValueError("nonunique or non-group-four rollout rows")
        seen.update(by_episode)
        groups.extend(by_episode.values())
    if len(groups) != 256:
        raise ValueError("incomplete training group coverage")
    table = {name: analyze(groups, name)
             for name in ("raw_failure", "stop_only", "signed_timeout")}
    old = table["raw_failure"]["ranking"]["large_gap_pairs"]
    signed = table["signed_timeout"]["ranking"]["large_gap_pairs"]
    signed_groups = table["signed_timeout"]
    gate = {
        "all_failure_group_variance_at_least_80pct":
            signed_groups["all_failure_groups_with_return_variance"] >=
            .8 * signed_groups["all_failure_groups"],
        "large_gap_near_ranking_no_worse_than_raw":
            signed["win_rate_excluding_ties"] is not None and
            old["win_rate_excluding_ties"] is not None and
            signed["win_rate_excluding_ties"] >=
            old["win_rate_excluding_ties"] - 1e-8,
        "success_priority_preserved":
            signed_groups["success_priority_violations"] == 0}
    result = {"schema": "signed_timeout_train_counterfactual_v1",
              "interpretation": "Train-scene counterfactual reward arithmetic on trajectories generated by the failed failure-only policy. It does not predict a newly trained policy or held-out navigation; no audit/val-unseen label is read.",
              "steps": 64, "group_size": 4, "episode_groups": len(groups),
              "formulas": {"raw_failure": "raw frozen bonus on every failure",
                           "stop_only": "max(0,2*raw-1) on unsuccessful voluntary STOP; else 0",
                           "signed_timeout": "max(0,2*raw-1) on unsuccessful STOP; raw-1 on timeout; -1 on format failure"},
              "table": table, "train_only_gate": gate,
              "eligible_for_later_online_screen": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
