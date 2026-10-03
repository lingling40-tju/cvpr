"""Exploratory confidence-gated Qwen group reward diagnostic.

The threshold is the upper median absolute same-mode failure-pair margin on
the frozen fit partition. The diagnostic was conceived after related cached
development results had been inspected, so all results remain exploratory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path


MODES = {"stopped but goal not reached.", "number of turns exceeded."}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_part(root: Path, groups: dict, policy: dict, part: str):
    scores = {}
    for shard in range(4):
        path = root / f"terminal_{part}_shard{shard}.json"
        source = json.loads(path.read_text())
        if source["schema"] != "qwen3_policy_terminal_shard_v1" or source["part"] != part:
            raise ValueError(f"teacher shard changed: {path}")
        for group in source["groups"]:
            for route in group["routes"]:
                rid = route["record_id"]
                if rid in scores:
                    raise ValueError(f"duplicate teacher route: {rid}")
                scores[rid] = float(route["states"][-1]["correct_margin_average"])
                if not math.isfinite(scores[rid]):
                    raise ValueError(f"nonfinite teacher score: {rid}")
    selected = {(row["seed"], str(row["episode_id"]))
                for row in policy["selected"][part]}
    by_group = defaultdict(list)
    for row in groups["selected"][part]:
        key = row["seed"], str(row["episode_id"])
        if key in selected:
            rid = f"s{key[0]}_e{key[1]}_v{row['variant']}"
            by_group[key].append({"record_id": rid,
                                  "score": scores[rid],
                                  "mode": row["terminal_mode"],
                                  "distance": float(row[
                                      "terminal_distance_m_for_replay_audit_only"])})
    if set(by_group) != selected or any(len(rows) != 4 for rows in by_group.values()):
        raise ValueError("selected group coverage changed")
    analysis = json.loads((root / f"terminal_{part}_analysis.json").read_text())
    outcomes = [scores[row["success_record"]] - scores[row["failure_record"]]
                for row in analysis["paired_rows"]]
    all_failure = {key: rows for key, rows in by_group.items()
                   if all(row["mode"] != "successfully reached the goal." for row in rows)}
    pairs = []
    for key, rows in all_failure.items():
        for left, right in itertools.combinations(rows, 2):
            if left["mode"] != right["mode"] or left["mode"] not in MODES:
                continue
            if abs(left["distance"] - right["distance"]) < 1.5:
                continue
            better, worse = (left, right) if left["distance"] < right["distance"] else (right, left)
            pairs.append({"key": key, "better": better["record_id"],
                          "worse": worse["record_id"],
                          "score_difference": better["score"] - worse["score"]})
    expected = {"fit": (33, 80, 71), "development": (None, 16, 19)}[part]
    if (expected[0] is not None and len(all_failure) != expected[0]) or \
            len(pairs) != expected[1] or len(outcomes) != expected[2]:
        raise ValueError("frozen pair denominators changed")
    return all_failure, pairs, outcomes


def gated_votes(rows: list[dict], threshold: float) -> dict[str, float]:
    votes = {row["record_id"]: 0.0 for row in rows}
    for mode in MODES:
        members = [row for row in rows if row["mode"] == mode]
        if len(members) < 2:
            continue
        for left, right in itertools.combinations(members, 2):
            difference = left["score"] - right["score"]
            if abs(difference) < threshold or difference == 0:
                continue
            sign = 1.0 if difference > 0 else -1.0
            votes[left["record_id"]] += sign
            votes[right["record_id"]] -= sign
        scale = 2 * (len(members) - 1)
        for item in members:
            votes[item["record_id"]] /= scale
        if abs(sum(votes[item["record_id"]] for item in members)) > 1e-9:
            raise ValueError("confidence votes are not zero-sum")
    if any(not math.isfinite(value) or abs(value) > 0.5 for value in votes.values()):
        raise ValueError("invalid bounded vote")
    return votes


def summary(groups: dict, pairs: list[dict], outcomes: list[float], threshold: float):
    rewards = {key: gated_votes(rows, threshold) for key, rows in groups.items()}
    eligible = [pair for pair in pairs if abs(pair["score_difference"]) >= threshold]
    active = [pair for pair in pairs if rewards[pair["key"]][pair["better"]] !=
              rewards[pair["key"]][pair["worse"]]]
    reliable_outcomes = [margin for margin in outcomes if abs(margin) >= threshold]
    return {
        "failure_pairs": len(pairs),
        "raw_correct": sum(pair["score_difference"] > 0 for pair in pairs),
        "high_confidence_pairs": len(eligible),
        "high_confidence_correct": sum(pair["score_difference"] > 0 for pair in eligible),
        "gated_reward_ordered_pairs": len(active),
        "gated_reward_correct": sum(
            rewards[pair["key"]][pair["better"]] >
            rewards[pair["key"]][pair["worse"]] for pair in active),
        "all_failure_groups": len(groups),
        "reward_active_groups": sum(any(value != 0 for value in gated_votes(rows, threshold).values())
                                    for rows in groups.values()),
        "success_failure_pairs": len(outcomes),
        "success_failure_raw_correct": sum(margin > 0 for margin in outcomes),
        "success_failure_high_confidence_pairs": len(reliable_outcomes),
        "success_failure_high_confidence_correct": sum(
            margin > 0 for margin in reliable_outcomes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy_path = args.root / "policy_manifest.json"
    policy = json.loads(policy_path.read_text())
    groups = json.loads(args.group_manifest.read_text())
    if groups["schema"] != "policy_group_relative_manifest_v1" or \
            policy["schema"] != "qwen3_policy_route_manifest_v1" or \
            policy["source_sha256"]["group_manifest"] != sha(args.group_manifest):
        raise ValueError("source manifest mismatch")
    parts = {part: load_part(args.root, groups, policy, part)
             for part in ("fit", "development")}
    fit_pairs = parts["fit"][1]
    threshold = sorted(abs(pair["score_difference"]) for pair in fit_pairs)[
        len(fit_pairs) // 2]
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid fit-only confidence threshold")
    report = {
        "schema": "qwen3_group_confidence_gate_diagnostic_v1",
        "threshold": {"method": "upper median absolute same-mode failure-pair margin on fit",
                      "value": threshold},
        "source_sha256": {"group_manifest": sha(args.group_manifest),
                          "policy_manifest": sha(policy_path),
                          **{f"terminal_{part}_analysis": sha(
                              args.root / f"terminal_{part}_analysis.json")
                             for part in parts},
                          **{f"terminal_{part}_shard{shard}": sha(
                              args.root / f"terminal_{part}_shard{shard}.json")
                             for part in parts for shard in range(4)}},
        "parts": {part: summary(*data, threshold) for part, data in parts.items()},
        "interpretation": "Post hoc reused train-scene diagnostic. The threshold is computed from fit pairs, but related development results had been inspected before this idea. No online reward or val-unseen navigation evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
