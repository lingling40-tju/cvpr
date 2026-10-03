"""One locked scene-heldout audit of the selected history-grounding LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part
from train_history_grounding_lora import evaluate, stripped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("locked audit already exists; do not rescore for tuning")
    report = json.loads(args.development.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if report["schema"] != "history_grounding_lora_development_v1" or \
            checkpoint["schema"] != "history_grounding_lora_seed11_v1" or \
            checkpoint["selected_step"] != report["selected_step"] or \
            checkpoint["source_sha256"] != report["source_sha256"] or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            checkpoint["development_stop_threshold"] != \
                report["development_stop_threshold"]["threshold"]:
        raise ValueError("development/checkpoint source mismatch")
    data = load_part("audit", args.scene_split, args.expert_manifest,
                     args.expert_labels, args.expert_root,
                     args.policy_manifest, args.policy_root, args.policy_audit)
    if {"scene_split": data["scene_split_sha256"],
        "expert_manifest": data["expert_manifest_sha256"],
        "policy_manifest": data["policy_manifest_sha256"]} != \
            report["source_sha256"]:
        raise ValueError("audit split/source hash mismatch")
    processor, model, head, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    head.load_state_dict(checkpoint["head"])
    metrics = evaluate(processor, model, head, data, small=False)
    threshold = checkpoint["development_stop_threshold"]
    negatives = metrics["negative_logits"]
    positives = metrics["positive_logits"]
    fpr = sum(x >= threshold for x in negatives) / len(negatives)
    recall = sum(x >= threshold for x in positives) / len(positives)
    gate = {
        "at_least_100_distinct_expert_trajectories_each_stop_class":
            len({row["record"]["trajectory_id"] for row in data["expert"]}) >= 100,
        "at_least_100_progress_pairs":
            metrics["forward_pairs"] + metrics["regression_pairs"] >= 100,
        "at_least_100_regression_pairs": metrics["regression_pairs"] >= 100,
        "progress_pair_accuracy_at_least_0_75":
            metrics["balanced_progress_accuracy"] >= .75,
        "regression_accuracy_at_least_0_60":
            metrics["regression_accuracy"] >= .60,
        "instruction_swap_accuracy_at_least_0_75":
            metrics["instruction_swap_accuracy"] >= .75,
        "stop_auc_at_least_0_80": metrics["stop_auc"] >= .80,
        "false_stop_rate_at_most_0_10": fpr <= .10,
        "stop_recall_at_least_0_50": recall >= .50,
    }
    result = {"schema": "history_grounding_lora_locked_audit_v1",
              "checkpoint_sha256": digest(args.checkpoint),
              "development_sha256": digest(args.development),
              "selected_step": checkpoint["selected_step"],
              "audit_scenes": len(data["scenes"]),
              "audit_expert_trajectories": len(data["expert"]),
              "audit_policy_trajectories": len(data["policy"]),
              "audit_policy_unique_episodes": len({
                  row["record"]["episode_id"] for row in data["policy"]}),
              "metrics": stripped(metrics),
              "development_selected_stop_threshold": threshold,
              "false_stop_rate": fpr, "stop_recall": recall,
              "predeclared_gate": gate,
              "eligible_for_group4_rl_pilot": all(gate.values()),
              "interpretation": "Train-scene locked representation audit only; no navigation gain measured."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
