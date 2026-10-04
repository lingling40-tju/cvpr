"""CPU-only coverage gate on the audited seed-11 n=4 exact512 oracle rollout.

This inventories labels before any new RGB replay or representation fit.
The oracle return is privileged train supervision, never a policy input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preflight_group_future_advantage import digest, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("rollout", "dataset", "scene-split", "train-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.train_audit.read_text())
    if (audit.get("schema") != "oracle_turnwise_exact512_scale_train_audit_v1"
            or audit.get("seed") != 11 or audit.get("steps") != 128
            or audit.get("group_size") != 4
            or audit.get("unique_train_episodes") != 512
            or audit.get("same_row_candidate_control") is not True
            or audit.get("nonzero_actor_gradient_steps", {}).get("candidate") != 128):
        raise ValueError("seed-11 paired training audit is incomplete")
    report = inventory(args.rollout, args.dataset, args.scene_split,
                       expected_rollout_sha=None, expected_steps=128,
                       expected_episodes=512, include_same_mode=True)
    report["schema"] = "group_future_advantage_exact512_preflight_v1"
    report["validated_train_audit_sha256"] = digest(args.train_audit)
    report["train_seed"] = 11
    report["group_size"] = 4
    report["steps"] = 128
    report["unique_train_episodes"] = 512
    # Fix the minimum before reading exact512 labels. Group size is counted
    # by episode, not by correlated trajectory-pair count.
    checks = {}
    for anchor in ("3", "6"):
        fit = report["parts"]["fit"]["anchors"][anchor]
        dev = report["parts"]["development"]["anchors"][anchor]
        checks[f"fit_anchor{anchor}_groups_at_least_150"] = (
            fit["episode_groups"] >= 150)
        checks[f"dev_anchor{anchor}_groups_at_least_35"] = (
            dev["episode_groups"] >= 35)
        checks[f"dev_anchor{anchor}_pairs_at_least_120"] = (
            dev["pairs"] >= 120)
        fit_same = report["parts"]["fit"]["same_terminal_mode_anchors"][anchor]
        dev_same = report["parts"]["development"]["same_terminal_mode_anchors"][anchor]
        checks[f"fit_anchor{anchor}_same_mode_groups_at_least_100"] = (
            fit_same["episode_groups"] >= 100)
        checks[f"dev_anchor{anchor}_same_mode_groups_at_least_25"] = (
            dev_same["episode_groups"] >= 25)
        checks[f"dev_anchor{anchor}_same_mode_pairs_at_least_60"] = (
            dev_same["pairs"] >= 60)
    report["coverage_checks"] = checks
    report["enough_coverage_for_fit_preparation"] = all(checks.values())
    report["interpretation"] = (
        "Audited seed-11 train-scene label coverage only. A pass permits "
        "prospective RGB-replay and scene-disjoint model protocol preparation, "
        "not a learned reward or navigation claim."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage_checks": checks,
                      "enough_coverage_for_fit_preparation":
                      report["enough_coverage_for_fit_preparation"]}))


if __name__ == "__main__":
    main()
