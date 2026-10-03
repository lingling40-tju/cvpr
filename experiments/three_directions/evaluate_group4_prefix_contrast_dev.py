"""Evaluate a fixed encoder pilot on scene-disjoint development states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics
from fit_group4_expert_prefix_value import expert_data, expert_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--previous-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    report = json.loads(args.training_report.read_text())
    frozen = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    previous = json.loads(args.previous_development.read_text())
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    source_id = digest(args.checkpoint)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            trained["schema"] != "group4_prefix_contrast_lora_v1" or \
            trained["source_sha256"] != report["source_sha256"] or \
            report["schema"] != "group4_prefix_contrast_lora_training_v1" or \
            report["smoke"] or report["microsteps"] != 512 or \
            trained["source_sha256"]["group_manifest"] != group_sha or \
            trained["source_sha256"]["expert_manifest"] != expert_sha or \
            trained["source_sha256"]["frozen_weights"] != digest(args.frozen_weights) or \
            frozen["schema"] != "group4_joint_value_weights_v1" or \
            previous["schema"] != "group4_expert_prefix_value_development_v1":
        raise ValueError("frozen development source mismatch")
    for kind, manifest, root, shards in (
            ("policy", group, args.group_state_root, 1),
            ("expert", expert, args.expert_state_root, 3)):
        rows = manifest["selected"]["development"]
        for shard in range(shards):
            summary = json.loads((root / "development" /
                                  f"summary_shard{shard}of{shards}.json").read_text())
            if summary["schema"] != "group4_prefix_contrast_dev_cache_shard_v1" or \
                    summary["kind"] != kind or summary["shard"] != shard or \
                    summary["shards"] != shards or \
                    summary["requested"] != len(rows[shard::shards]) or \
                    summary["completed"] != len(rows[shard::shards]) or \
                    summary["manifest_sha256"] != digest(
                        args.group_manifest if kind == "policy" else args.expert_manifest) or \
                    summary["source_id"] != source_id:
                raise ValueError(f"incomplete development shard {kind}/{shard}")
    coverage, group_pairs = load_part(
        group, "development", args.group_turn_root,
        args.group_state_root, group_sha, source_id)
    expert_rows, expert_difference = expert_data(
        expert, "development", args.expert_state_root, expert_sha, source_id)
    if coverage["trajectories"] != 160 or \
            coverage["matched_pairs"] != 103 or \
            coverage["matched_groups"] != 18 or \
            len(expert_rows) != 303 or \
            len({row["scene_id"] for row in expert_rows}) != 8:
        raise ValueError("development metric coverage mismatch")
    scale = frozen["scale"].float()
    vector = frozen["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all()) or \
            not bool(torch.isfinite(vector).all()):
        raise ValueError("bad frozen readout")
    with torch.no_grad():
        outcome_x = F.normalize(
            torch.stack([row["difference"] for row in group_pairs]) / scale,
            dim=1)
        expert_x = F.normalize(expert_difference / scale, dim=1)
        outcome_scores = outcome_x @ vector
        expert_scores = expert_x @ vector
    outcome = metrics(group_pairs, outcome_scores)
    instruction = expert_metrics(expert_rows, expert_scores)
    old = previous["development"]["old_outcome_only_readout"]
    if old["outcome"]["correct"] != 74 or old["expert"]["correct"] != 196:
        raise ValueError("old development control changed")
    gate = {"outcome_accuracy_at_least_0_70": outcome["accuracy"] >= .70,
            "outcome_group_macro_at_least_0_70":
                outcome["group_macro_accuracy"] >= .70,
            "outcome_no_more_than_two_pairs_below_old_joint":
                outcome["correct"] >= old["outcome"]["correct"] - 2,
            "expert_prefix_accuracy_at_least_0_75":
                instruction["accuracy"] >= .75,
            "expert_prefix_scene_macro_at_least_0_70":
                instruction["scene_macro_accuracy"] >= .70}
    result = {"schema": "group4_prefix_contrast_lora_development_v1",
              "interpretation": "scene-disjoint development of fixed encoder pilot; no navigation or blind-audit result",
              "source_sha256": {"group_manifest": group_sha,
                                "expert_manifest": expert_sha,
                                "checkpoint": source_id,
                                "training_report": digest(args.training_report),
                                "frozen_weights": digest(args.frozen_weights),
                                "previous_development": digest(args.previous_development)},
              "coverage": {"group": coverage,
                           "expert_prefix_pairs": len(expert_rows)},
              "outcome": outcome, "expert_prefix_instruction": instruction,
              "previous_frozen_joint": old,
              "predeclared_gate": gate,
              "eligible_for_policy_swap_development": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
