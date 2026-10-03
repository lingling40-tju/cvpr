"""Audit a fixed prefix of current on-policy train rollouts for score transfer.

Distance is used only for this train-scene diagnostic. No reward, model, or
evaluation manifest is selected from these labels. The 5.5 threshold was
fixed from the older fit cache before these rollouts were produced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path


GAP = 5.5
MODES = ("stopped but goal not reached.", "number of turns exceeded.")


def read_prefix(path: Path, steps: int):
    with path.open("rb") as stream:
        lines = [stream.readline() for _ in range(steps)]
    if len(lines) != steps or any(not line for line in lines):
        raise ValueError(f"rollout prefix shorter than {steps}: {path}")
    return [json.loads(line) for line in lines], hashlib.sha256(b"".join(lines)).hexdigest()


def votes(items: list[dict]) -> dict[int, float]:
    value = {id(item): 0.0 for item in items}
    for mode in MODES:
        members = [item for item in items if item["end_reason"] == mode]
        if len(members) < 2:
            continue
        for left, right in itertools.combinations(members, 2):
            difference = float(left["fused_reward"]["raw"]) - \
                         float(right["fused_reward"]["raw"])
            if not math.isfinite(difference):
                raise ValueError("nonfinite teacher difference")
            if abs(difference) < GAP:
                continue
            amount = (1 if difference > 0 else -1) / (2 * (len(members) - 1))
            value[id(left)] += amount
            value[id(right)] -= amount
        if abs(sum(value[id(item)] for item in members)) > 1e-9:
            raise ValueError("hypothetical confidence vote not zero-sum")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rollout", type=Path, required=True)
    parser.add_argument("--control-rollout", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        raise ValueError("expected 1..64 training steps")
    candidate, candidate_sha = read_prefix(args.candidate_rollout, args.steps)
    control, control_sha = read_prefix(args.control_rollout, args.steps)
    count = Counter()
    seen = set()
    for step, (row, base) in enumerate(zip(candidate, control), 1):
        if row["step"] != step or base["step"] != step or \
                len(row["info"]) != 16 or len(base["info"]) != 16:
            raise ValueError("step or rollout-count mismatch")
        groups = defaultdict(list)
        for item in row["info"]:
            groups[str(item["episode_id"])].append(item)
        base_ids = Counter(str(item["episode_id"]) for item in base["info"])
        if set(groups) != set(base_ids) or seen.intersection(groups) or \
                set(map(len, groups.values())) != {4} or \
                set(base_ids.values()) != {4}:
            raise ValueError("candidate/control training episodes do not match")
        seen.update(groups)
        for items in groups.values():
            count["groups"] += 1
            if any(item["task_success"] for item in items):
                continue
            count["all_failure_groups"] += 1
            for item in items:
                if item["fused_reward"]["status"] != "ok":
                    raise ValueError("all-failure group missing teacher score")
            gated = votes(items)
            count["gated_active_groups"] += any(value != 0 for value in gated.values())
            for left, right in itertools.combinations(items, 2):
                if left["end_reason"] != right["end_reason"] or \
                        left["end_reason"] not in MODES:
                    continue
                da = float(left["distance_to_goal"])
                db = float(right["distance_to_goal"])
                if not math.isfinite(da) or not math.isfinite(db):
                    raise ValueError("nonfinite simulator distance label")
                if abs(da - db) < 1.5:
                    continue
                better, worse = (left, right) if da < db else (right, left)
                difference = float(better["fused_reward"]["raw"]) - \
                             float(worse["fused_reward"]["raw"])
                current = (float(better["reward_components"]["qwen_group_ordinal"]) -
                           float(worse["reward_components"]["qwen_group_ordinal"]))
                count["eligible_pairs"] += 1
                count["raw_correct"] += difference > 0
                count["current_rank_correct"] += current > 0
                if abs(difference) >= GAP:
                    count["high_gap_pairs"] += 1
                    count["high_gap_correct"] += difference > 0
                hypothetical = gated[id(better)] - gated[id(worse)]
                if hypothetical != 0:
                    count["gated_ordered_pairs"] += 1
                    count["gated_correct"] += hypothetical > 0
    if count["groups"] != 4 * args.steps or len(seen) != 4 * args.steps or \
            count["eligible_pairs"] == 0:
        raise ValueError("incomplete train-scene diagnostic coverage")
    report = {"schema": "qwen_confident_onpolicy_train_diagnostic_v1",
              "steps": args.steps, "group_size": 4,
              "candidate_prefix_sha256": candidate_sha,
              "control_prefix_sha256": control_sha,
              "min_score_gap": GAP,
              "counts": dict(count),
              "accuracies": {
                  "raw_on_eligible": count["raw_correct"] / count["eligible_pairs"],
                  "current_rank_on_eligible": count["current_rank_correct"] / count["eligible_pairs"],
                  "high_gap_on_retained": count["high_gap_correct"] / count["high_gap_pairs"]
                  if count["high_gap_pairs"] else None,
                  "hypothetical_gated_on_ordered": count["gated_correct"] /
                  count["gated_ordered_pairs"] if count["gated_ordered_pairs"] else None},
              "interpretation": "Exploratory R2R train-scene on-policy ranking, not val-unseen navigation or a trained confidence-policy result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
