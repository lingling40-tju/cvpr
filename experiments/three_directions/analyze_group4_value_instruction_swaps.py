"""Merge paired natural wrong-goal instruction scores without model tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swaps", type=Path, required=True)
    parser.add_argument("--locked-audit", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    swaps = json.loads(args.swaps.read_text())
    locked = json.loads(args.locked_audit.read_text())
    pieces = [json.loads(path.read_text()) for path in args.shard]
    if len(pieces) != 4 or {part["shard"] for part in pieces} != set(range(4)):
        raise ValueError("expected four distinct extraction shards")
    groups = []
    for part in pieces:
        if part["schema"] != "group4_value_instruction_swap_shard_v1" or \
                part["shards"] != 4 or \
                part["manifest_sha256"] != digest(args.manifest) or \
                part["swaps_sha256"] != digest(args.swaps) or \
                part["locked_audit_sha256"] != digest(args.locked_audit) or \
                part["weights_sha256"] != locked["weights_sha256"] or \
                part["trajectory_count"] != 4 * part["group_count"] or \
                part["pair_count"] != sum(len(g["pairs"]) for g in part["groups"]):
            raise ValueError(f"invalid shard {part.get('shard')}")
        groups.extend(part["groups"])
    expected_groups = {f"s{row['seed']}_e{row['episode_id']}" for row in
                       source["selected"]["audit"]}
    if len(groups) != len(expected_groups) or \
            {group["group_id"] for group in groups} != expected_groups:
        raise ValueError("incomplete audit group coverage")
    pairs = [pair for group in groups for pair in group["pairs"]]
    original_correct = sum(pair["original_margin"] > 0 for pair in pairs)
    swapped_correct = sum(pair["swapped_margin"] > 0 for pair in pairs)
    active = [group for group in groups if group["pairs"]]
    if len(pairs) != locked["metrics"]["pairs"] or \
            original_correct != locked["metrics"]["correct"] or \
            len(active) != locked["metrics"]["groups"]:
        raise ValueError("original-instruction recomputation differs from locked audit")
    group_macro = sum(sum(p["original_margin"] > 0 for p in g["pairs"]) /
                      len(g["pairs"]) for g in active) / len(active)
    if abs(group_macro - locked["metrics"]["group_macro_accuracy"]) > 1e-9:
        raise ValueError("group-macro recomputation mismatch")
    group_margin_drop = [sum(p["original_margin"] - p["swapped_margin"]
                             for p in g["pairs"]) / len(g["pairs"])
                         for g in active]
    groups_with_drop = sum(value > 0 for value in group_margin_drop)
    diffs = [sum(p["original_margin"] > 0 for p in g["pairs"]) / len(g["pairs"]) -
             sum(p["swapped_margin"] > 0 for p in g["pairs"]) / len(g["pairs"])
             for g in active]
    rng = random.Random(11)
    clustered = []
    for _ in range(5000):
        chosen = [rng.choice(active) for _ in active]
        n = sum(len(g["pairs"]) for g in chosen)
        delta = sum(sum((p["original_margin"] > 0) -
                        (p["swapped_margin"] > 0) for p in g["pairs"])
                    for g in chosen) / n
        clustered.append(delta)
    clustered.sort()
    same_start = [g for g in active if g["same_start_swap"]]
    gate = {"all_131_pairs_covered": len(pairs) == 131,
            "all_22_mixed_groups_covered": len(active) == 22,
            "swapped_accuracy_drop_at_least_0_10":
                (original_correct - swapped_correct) / len(pairs) >= .10,
            "group_margin_drop_fraction_at_least_0_60":
                groups_with_drop / len(active) >= .60}
    result = {"schema": "group4_value_instruction_swap_analysis_v1",
              "manifest_sha256": digest(args.manifest),
              "swaps_sha256": digest(args.swaps),
              "locked_audit_sha256": digest(args.locked_audit),
              "shard_sha256": [digest(path) for path in args.shard],
              "all_groups": len(groups), "active_groups": len(active),
              "pairs": len(pairs),
              "original_correct": original_correct,
              "original_accuracy": original_correct / len(pairs),
              "swapped_correct": swapped_correct,
              "swapped_accuracy": swapped_correct / len(pairs),
              "accuracy_drop": (original_correct - swapped_correct) / len(pairs),
              "group_margin_drop": groups_with_drop,
              "group_margin_drop_fraction": groups_with_drop / len(active),
              "group_cluster_bootstrap_accuracy_drop_95":
                  [clustered[125], clustered[4874]],
              "same_start_active_groups": len(same_start),
              "same_start_pairs": sum(len(g["pairs"]) for g in same_start),
              "predeclared_gate": gate,
              "eligible_for_group4_reward_wiring": all(gate.values()),
              "interpretation": "instruction dependence diagnostic with natural same-scene wrong goals, not an independent semantic or navigation validation"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
