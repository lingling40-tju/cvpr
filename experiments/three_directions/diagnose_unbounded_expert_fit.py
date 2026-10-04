"""Diagnose fit versus scene-disjoint development for the selected head.

Run only after the frozen full fit has finished. This is a train-scene
diagnostic and cannot be used as an independent model-selection gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import digest, load_model
from train_balanced_change_lora import load_expanded_fit
import train_policy_progress_lora as prior
from train_unbounded_expert_cross_goal_potential_lora import (
    UnboundedPotentialHead, load_new_fit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "expanded-manifest", "expanded-root",
                 "expanded-audit", "render-manifest", "render-root",
                 "rgb-audit", "cross-manifest", "cross-root",
                 "cross-audit", "model", "fit-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.fit_output / "development.json").read_text())
    if (report["schema"] != "unbounded_expert_cross_goal_potential_lora_development_v1"
            or report["microsteps"] != 1500 or report["selected_step"] not in
            (250, 500, 1000, 1500)):
        raise ValueError("full fit report missing or invalid")
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit, _ = load_expanded_fit(common, args.expanded_manifest,
                               args.expanded_root, args.expanded_audit)
    new_items, _, _ = load_new_fit(args, fit)
    fit["policy"].extend(new_items)
    expected = report["source_sha256"]
    sources = {
        "scene_split": digest(args.scene_split),
        "expert_manifest": digest(args.expert_manifest),
        "old_policy_manifest": digest(args.policy_manifest),
        "expanded_policy_manifest": digest(args.expanded_manifest),
        "render_manifest": digest(args.render_manifest),
        "rgb_audit": digest(args.rgb_audit),
        "cross_manifest": digest(args.cross_manifest),
        "cross_audit": digest(args.cross_audit),
        "model_config": digest(args.model / "config.json"),
    }
    if sources != expected:
        raise ValueError("source hashes differ from full fit")
    checkpoint_path = args.fit_output / "adapter_head.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["schema"] != "unbounded_expert_cross_goal_potential_lora_v1"
            or checkpoint["source_sha256"] != sources
            or checkpoint["selected_step"] != report["selected_step"]):
        raise ValueError("selected adapter provenance mismatch")
    processor, model, bounded_head, _ = load_model(args.model)
    head = UnboundedPotentialHead(bounded_head)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    head.load_state_dict(checkpoint["head"])
    fit_metrics = prior.evaluate(processor, model, head, fit, small=True)
    dev_metrics = report["selected_small_development"]
    result = {
        "schema": "unbounded_expert_fit_development_diagnostic_v1",
        "interpretation": "Fit rows were used in training; development is reused for checkpoint choice; no audit, RL, or navigation result",
        "selected_step": checkpoint["selected_step"],
        "source_sha256": sources,
        "adapter_sha256": digest(checkpoint_path),
        "fit_small": fit_metrics,
        "selected_development_small": dev_metrics,
        "fit_minus_development": {
            "balanced_direction_accuracy": fit_metrics["balanced_direction_accuracy"]
            - dev_metrics["balanced_direction_accuracy"],
            "instruction_gain_preference": fit_metrics["instruction_gain_preference"]
            - dev_metrics["instruction_gain_preference"],
            "forward_recall_at_stationary_threshold":
            fit_metrics["forward_recall_at_stationary_threshold"]
            - dev_metrics["forward_recall_at_stationary_threshold"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["fit_minus_development"], sort_keys=True))


if __name__ == "__main__":
    main()
