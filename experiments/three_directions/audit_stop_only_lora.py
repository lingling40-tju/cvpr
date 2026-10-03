"""One-time locked STOP-only audit of the frozen history-grounded LoRA.

The progress head is disregarded. The development-selected threshold is
fixed, and all positive/negative subclasses are scored once on held-out
R2R-train scenes. Existing output is never overwritten for retuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part, score
from stop_history_head import auc


def fpr(values: list[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / len(values)


def recall(values: list[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / len(values)


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
        raise ValueError("STOP-only locked audit already exists")
    report = json.loads(args.development.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    threshold_report = report["development_stop_threshold"]
    threshold = float(threshold_report["threshold"])
    if (report["schema"] != "history_grounding_lora_development_v1" or
            checkpoint["schema"] != "history_grounding_lora_seed11_v1" or
            checkpoint["selected_step"] != report["selected_step"] or
            checkpoint["source_sha256"] != report["source_sha256"] or
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or
            checkpoint["development_stop_threshold"] != threshold or
            report["development"]["stop_auc"] < .80 or
            report["development"]["instruction_swap_accuracy"] < .75 or
            threshold_report["development_fpr"] > .10 or
            threshold_report["development_recall"] < .50):
        raise ValueError("frozen development STOP eligibility failed")
    data = load_part("audit", args.scene_split, args.expert_manifest,
                     args.expert_labels, args.expert_root,
                     args.policy_manifest, args.policy_root, args.policy_audit)
    if ({"scene_split": data["scene_split_sha256"],
         "expert_manifest": data["expert_manifest_sha256"],
         "policy_manifest": data["policy_manifest_sha256"]} !=
            report["source_sha256"]):
        raise ValueError("audit source hash mismatch")
    torch.set_num_threads(6)
    processor, model, head, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    head.load_state_dict(checkpoint["head"])
    model.eval()
    head.eval()
    classes = {name: [] for name in
               ("expert_goal", "expert_start", "expert_wrong",
                "policy_goal", "policy_far")}
    swaps = 0
    started = time.time()
    with torch.inference_mode():
        for index, item in enumerate(data["expert"], 1):
            record = item["record"]
            at_goal, _ = score(processor, model, head, item,
                               len(record["turns"]))
            at_start, _ = score(processor, model, head, item, 0)
            classes["expert_goal"].append(float(at_goal))
            classes["expert_start"].append(float(at_start))
            if item["safe_swap"]:
                wrong, _ = score(processor, model, head, item,
                                 len(record["turns"]),
                                 record["wrong_instruction"])
                classes["expert_wrong"].append(float(wrong))
                swaps += int(float(at_goal) > float(wrong))
            if index % 25 == 0:
                print(f"expert {index}/{len(data['expert'])} "
                      f"elapsed_s={time.time()-started:.1f}", flush=True)
        for index, item in enumerate(data["policy"], 1):
            distance = item["distances"][-1]
            if 3.0 < distance < 3.5:
                continue
            terminal, _ = score(processor, model, head, item,
                                len(item["record"]["turns"]))
            key = "policy_goal" if distance <= 3.0 else "policy_far"
            classes[key].append(float(terminal))
            if index % 50 == 0:
                print(f"policy {index}/{len(data['policy'])} "
                      f"elapsed_s={time.time()-started:.1f}", flush=True)
    if any(not values for values in classes.values()):
        raise ValueError("missing STOP audit class")
    positives = classes["expert_goal"] + classes["policy_goal"]
    negatives = classes["expert_start"] + classes["expert_wrong"] + \
                classes["policy_far"]
    metrics = {"stop_auc": auc(torch.tensor(positives),
                                torch.tensor(negatives)),
               "pooled_fpr": fpr(negatives, threshold),
               "pooled_recall": recall(positives, threshold),
               "instruction_swap_accuracy": swaps / len(classes["expert_wrong"]),
               "wrong_instruction_fpr": fpr(classes["expert_wrong"], threshold),
               "policy_far_fpr": fpr(classes["policy_far"], threshold),
               "policy_goal_recall": recall(classes["policy_goal"], threshold),
               "expert_start_fpr": fpr(classes["expert_start"], threshold),
               "expert_goal_recall": recall(classes["expert_goal"], threshold)}
    counts = {name: len(values) for name, values in classes.items()}
    unique_expert = len({item["record"]["trajectory_id"]
                         for item in data["expert"]})
    gate = {"expert_trajectories_at_least_100": unique_expert >= 100,
            "safe_swaps_at_least_100": counts["expert_wrong"] >= 100,
            "policy_goal_at_least_50": counts["policy_goal"] >= 50,
            "policy_far_at_least_100": counts["policy_far"] >= 100,
            "stop_auc_at_least_0_80": metrics["stop_auc"] >= .80,
            "instruction_swap_accuracy_at_least_0_75":
                metrics["instruction_swap_accuracy"] >= .75,
            "pooled_fpr_at_most_0_10": metrics["pooled_fpr"] <= .10,
            "pooled_recall_at_least_0_50": metrics["pooled_recall"] >= .50,
            "wrong_instruction_fpr_at_most_0_15":
                metrics["wrong_instruction_fpr"] <= .15,
            "policy_far_fpr_at_most_0_10":
                metrics["policy_far_fpr"] <= .10,
            "policy_goal_recall_at_least_0_50":
                metrics["policy_goal_recall"] >= .50}
    result = {"schema": "stop_only_lora_locked_audit_v1",
              "interpretation": "One-time train-scene STOP-only audit. Progress head discarded; no navigation gain measured.",
              "checkpoint_sha256": digest(args.checkpoint),
              "development_sha256": digest(args.development),
              "selected_step": checkpoint["selected_step"],
              "threshold": threshold,
              "audit_scenes": len(data["scenes"]),
              "expert_unique_trajectories": unique_expert,
              "policy_unique_episodes": len({item["record"]["episode_id"]
                                             for item in data["policy"]}),
              "counts": counts, "metrics": metrics,
              "predeclared_gate": gate,
              "eligible_for_group4_stop_pilot": all(gate.values()),
              "elapsed_seconds": time.time() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
