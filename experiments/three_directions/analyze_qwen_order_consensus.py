"""Exploratory test of A/B-order consensus as a reward confidence gate.

Uses existing frozen teacher scores and simulator labels only for
analysis. This is a post hoc diagnostic, not a registered selection
gate or a navigation result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path


MODES = {"stopped but goal not reached.", "number of turns exceeded."}


def counts(rows, score):
    summary = Counter()
    for better, worse in rows:
        a, b = score[better], score[worse]
        agrees = (a[0] - b[0]) * (a[1] - b[1]) > 0
        correct = a[2] > b[2]
        summary["all_pairs"] += 1
        summary["all_correct"] += correct
        summary["consensus_pairs"] += agrees
        summary["consensus_correct"] += agrees and correct
        summary["discordant_pairs"] += not agrees
        summary["discordant_correct"] += not agrees and correct
    return dict(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    policy = json.loads(args.policy_manifest.read_text())
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            policy["schema"] != "qwen3_policy_route_manifest_v1":
        raise ValueError("source schema mismatch")
    report = {"schema": "qwen3_order_consensus_diagnostic_v1",
              "interpretation": "Post hoc reused train-scene diagnostic; no reward selection or navigation claim.",
              "parts": {}}
    for part, expected_outcome, expected_failure in (
            ("fit", 71, 80), ("development", 19, 16)):
        score = {}
        for shard in range(4):
            data = json.loads((args.root /
                               f"terminal_{part}_shard{shard}.json").read_text())
            if data["schema"] != "qwen3_policy_terminal_shard_v1" or \
                    data["part"] != part:
                raise ValueError("frozen teacher shard mismatch")
            for item in data["groups"]:
                for route in item["routes"]:
                    state = route["states"][-1]
                    score[route["record_id"]] = (
                        float(state["correct_margin_first"]),
                        float(state["correct_margin_swapped"]),
                        float(state["correct_margin_average"]))
        analysis = json.loads((args.root /
                               f"terminal_{part}_analysis.json").read_text())
        outcome = [(row["success_record"], row["failure_record"])
                   for row in analysis["paired_rows"]]
        selected = {(row["seed"], str(row["episode_id"]))
                    for row in policy["selected"][part]}
        groups = defaultdict(list)
        for row in group["selected"][part]:
            key = row["seed"], str(row["episode_id"])
            if key in selected:
                groups[key].append(row)
        failure = []
        for (seed, eid), rows in groups.items():
            if len(rows) != 4 or any(
                    row["terminal_mode"] == "successfully reached the goal."
                    for row in rows):
                continue
            for left, right in itertools.combinations(rows, 2):
                if left["terminal_mode"] != right["terminal_mode"] or \
                        left["terminal_mode"] not in MODES:
                    continue
                dl = float(left["terminal_distance_m_for_replay_audit_only"])
                dr = float(right["terminal_distance_m_for_replay_audit_only"])
                if abs(dl - dr) < 1.5:
                    continue
                better, worse = (left, right) if dl < dr else (right, left)
                failure.append((f"s{seed}_e{eid}_v{better['variant']}",
                                f"s{seed}_e{eid}_v{worse['variant']}"))
        if len(outcome) != expected_outcome or len(failure) != expected_failure:
            raise ValueError("label opportunity denominator changed")
        report["parts"][part] = {
            "success_over_failure": counts(outcome, score),
            "same_mode_closer_failure": counts(failure, score)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["parts"], indent=2))


if __name__ == "__main__":
    main()
