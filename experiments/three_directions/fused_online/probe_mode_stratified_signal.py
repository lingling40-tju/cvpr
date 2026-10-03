"""Freeze a train-rollout signal screen for mode-stratified ordinal rewards."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from mode_stratified_reward import ELIGIBLE_MODES, mode_stratified_adjustments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hasher = hashlib.sha256()
    totals = defaultdict(int)
    by_mode = {mode: defaultdict(int) for mode in ELIGIBLE_MODES}
    for line in args.rollout.open("rb"):
        hasher.update(line)
        batch = json.loads(line)
        rows = batch["info"]
        assert len(rows) == 16
        _, ordinal, summary = mode_stratified_adjustments(rows, 4)
        for key, value in summary.items():
            totals[key] += value
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row["episode_id"])].append(index)
        for indices in groups.values():
            if any(rows[index]["task_success"] for index in indices):
                assert all(ordinal[index] == 0 for index in indices)
                continue
            for mode in ELIGIBLE_MODES:
                members = [index for index in indices
                           if rows[index]["end_reason"] == mode]
                if len(members) >= 2:
                    by_mode[mode]["groups_with_two_or_more"] += 1
                for offset, first in enumerate(members):
                    for second in members[offset + 1:]:
                        distance_delta = (rows[first]["distance_to_goal"] -
                                          rows[second]["distance_to_goal"])
                        if abs(distance_delta) < 1.5:
                            continue
                        by_mode[mode]["large_gap_pairs"] += 1
                        rank_delta = ordinal[first] - ordinal[second]
                        if rank_delta * distance_delta < 0:
                            by_mode[mode]["correct_large_gap_pairs"] += 1
                        elif rank_delta == 0:
                            by_mode[mode]["tied_large_gap_pairs"] += 1
    assert totals["matched_groups"] == 256
    assert totals["all_failure_groups"] == 166
    evaluated = sum(x["large_gap_pairs"] for x in by_mode.values())
    correct = sum(x["correct_large_gap_pairs"] for x in by_mode.values())
    assert evaluated > 0
    report = {
        "schema": "mode_stratified_ordinal_train_screen_v1",
        "interpretation": "Completed failure-only policy's train-scene rollouts, used as a frozen signal screen; no val-unseen or retrained-policy claim.",
        "rollout_sha256": hasher.hexdigest(),
        "group_size": 4,
        "counts": dict(totals),
        "by_terminal_mode": {mode: dict(row) for mode, row in by_mode.items()},
        "large_gap_same_mode_pairs": evaluated,
        "correct_large_gap_pairs": correct,
        "same_mode_accuracy": correct / evaluated,
        "predeclared_pilot_gate": {
            "rule": "at least 100 active all-failure groups and at least 60% correct among 1.5m-gap same-mode pairs",
            "pass": totals["ordinal_active_groups"] >= 100 and correct / evaluated >= .60,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
