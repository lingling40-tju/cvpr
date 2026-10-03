"""One-time model audit for the frozen group-four joint linear readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics
from fit_group4_joint_value import expert_data, expert_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-collection-audit", type=Path, required=True)
    parser.add_argument("--group-state-audit", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-state-audit", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    group_collection = json.loads(args.group_collection_audit.read_text())
    group_audit = json.loads(args.group_state_audit.read_text())
    expert_audit = json.loads(args.expert_state_audit.read_text())
    development = json.loads(args.development_report.read_text())
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            development["schema"] != "group4_joint_value_development_v1" or \
            not development["eligible_for_model_audit"] or \
            not all(development["predeclared_gate"].values()) or \
            weights["schema"] != "group4_joint_value_weights_v1" or \
            development["source_sha256"]["group_manifest"] != group_sha or \
            development["source_sha256"]["expert_manifest"] != expert_sha or \
            weights["source_sha256"] != development["source_sha256"] or \
            group_collection["manifest_sha256"] != group_sha or \
            group_audit["manifest_sha256"] != group_sha or \
            expert_audit["manifest_sha256"] != expert_sha or \
            group_audit["source_id"] != expert_audit["source_id"] or \
            group_audit["source_id"] != weights["encoder_source_id"]:
        raise ValueError("joint audit source or frozen model mismatch")
    coverage, outcome_pairs = load_part(group, "audit", args.group_turn_root,
                                        args.group_state_root, group_sha,
                                        group_audit["source_id"])
    expert_rows, expert_difference = expert_data(
        expert, "audit", args.expert_state_root, expert_sha,
        expert_audit["source_id"])
    if coverage["trajectories"] != group_audit["parts"]["audit"]["trajectories"] or \
            len(expert_rows) != expert_audit["parts"]["audit"]["expert_contrasts"]:
        raise ValueError("incomplete model audit cache")
    scale = weights["scale"].float()
    vector = weights["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all() and torch.isfinite(vector).all()):
        raise ValueError("bad frozen joint readout")
    with torch.no_grad():
        outcome_x = F.normalize(torch.stack([x["difference"] for x in outcome_pairs]) /
                                scale, dim=1)
        expert_x = F.normalize(expert_difference / scale, dim=1)
        outcome_scores = outcome_x @ vector
        expert_scores = expert_x @ vector
    outcome = metrics(outcome_pairs, outcome_scores)
    instruction = expert_metrics(expert_rows, expert_scores)
    gate = {"outcome_pairs_at_least_100": outcome["pairs"] >= 100,
            "outcome_groups_at_least_20": outcome["groups"] >= 20,
            "expert_pairs_at_least_100": instruction["pairs"] >= 100,
            "expert_scenes_at_least_8": instruction["scenes"] >= 8,
            "outcome_accuracy_at_least_0_70": outcome["accuracy"] >= .70,
            "outcome_group_macro_at_least_0_70":
                outcome["group_macro_accuracy"] >= .70,
            "expert_accuracy_at_least_0_75": instruction["accuracy"] >= .75,
            "expert_scene_macro_at_least_0_70":
                instruction["scene_macro_accuracy"] >= .70}
    result = {"schema": "group4_joint_value_locked_audit_v1",
              "interpretation": "one-time model-held-out R2R-train audit with research-wide scene reuse; no navigation improvement measured",
              "source_sha256": {
                  "group_manifest": group_sha,
                  "group_collection_audit": digest(args.group_collection_audit),
                  "group_state_audit": digest(args.group_state_audit),
                  "expert_manifest": expert_sha,
                  "expert_state_audit": digest(args.expert_state_audit),
                  "development_report": digest(args.development_report),
                  "weights": digest(args.weights)},
              "coverage": {"group": coverage,
                           "expert_pairs": len(expert_rows)},
              "outcome": outcome, "expert_instruction": instruction,
              "predeclared_gate": gate,
              "eligible_for_policy_instruction_recheck": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
