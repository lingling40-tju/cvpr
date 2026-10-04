"""One-time train-scene audit of the policy-prompt STOP representation.

The audit scenes were examined in previous method development. This is
therefore an exploratory transfer check, never independent verifier truth.
The script refuses to read them unless the predeclared development gate
passes and writes its result only once.
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


def rate(values: list[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "model", "checkpoint", "development",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("STOP hard-negative audit already exists")
    report = json.loads(args.development.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    if report["schema"] != "policy_stop_hardneg_lora_development_v1" or \
            checkpoint["schema"] != "policy_stop_hardneg_lora_seed11_v1" or \
            checkpoint["prompt_mode"] != "policy_multiturn_without_system" or \
            checkpoint["selected_step"] != report["selected_step"] or \
            checkpoint["source_sha256"] != report["source_sha256"] or \
            checkpoint["model_config_sha256"] != \
                digest(args.model / "config.json") or \
            checkpoint["development_stop_threshold"] != \
                report["development"]["threshold"]["threshold"]:
        raise ValueError("development/checkpoint identity mismatch")
    if not report["eligible_for_locked_audit"] or \
            not all(report["predeclared_gate"].values()):
        raise ValueError("development gate failed; audit remains unopened")
    data = load_part("audit", args.scene_split, args.expert_manifest,
                     args.expert_labels, args.expert_root,
                     args.policy_manifest, args.policy_root,
                     args.policy_audit)
    if {"scene_split": data["scene_split_sha256"],
        "expert_manifest": data["expert_manifest_sha256"],
        "policy_manifest": data["policy_manifest_sha256"]} != \
            report["source_sha256"]:
        raise ValueError("audit split/source hash mismatch")
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
            end = len(record["turns"])
            good, _ = score(processor, model, head, item, end,
                            include_system=False)
            start, _ = score(processor, model, head, item, 0,
                             include_system=False)
            classes["expert_goal"].append(float(good))
            classes["expert_start"].append(float(start))
            if item["safe_swap"]:
                wrong, _ = score(processor, model, head, item, end,
                                 record["wrong_instruction"],
                                 include_system=False)
                classes["expert_wrong"].append(float(wrong))
                swaps += int(float(good) > float(wrong))
            if index % 25 == 0:
                print(f"expert {index}/{len(data['expert'])} "
                      f"elapsed_s={time.time()-started:.1f}", flush=True)
        for index, item in enumerate(data["policy"], 1):
            distance = item["distances"][-1]
            if 3.0 < distance < 3.5:
                continue
            logit, _ = score(processor, model, head, item,
                             len(item["record"]["turns"]),
                             include_system=False)
            classes["policy_goal" if distance <= 3.0 else
                    "policy_far"].append(float(logit))
            if index % 50 == 0:
                print(f"policy {index}/{len(data['policy'])} "
                      f"elapsed_s={time.time()-started:.1f}", flush=True)
    if any(not values for values in classes.values()):
        raise ValueError("missing STOP audit class")
    positive = classes["expert_goal"] + classes["policy_goal"]
    negative = (classes["expert_start"] + classes["expert_wrong"] +
                classes["policy_far"])
    threshold = float(checkpoint["development_stop_threshold"])
    metrics = {"stop_auc": auc(torch.tensor(positive),
                                torch.tensor(negative)),
               "pooled_fpr": rate(negative, threshold),
               "pooled_recall": rate(positive, threshold),
               "instruction_swap_accuracy": swaps / len(classes["expert_wrong"]),
               "wrong_instruction_fpr": rate(classes["expert_wrong"], threshold),
               "policy_far_fpr": rate(classes["policy_far"], threshold),
               "policy_goal_recall": rate(classes["policy_goal"], threshold)}
    counts = {key: len(values) for key, values in classes.items()}
    gate = {"expert_trajectories_at_least_100":
                len({x["record"]["trajectory_id"] for x in data["expert"]}) >= 100,
            "expert_wrong_at_least_100": counts["expert_wrong"] >= 100,
            "policy_goal_at_least_50": counts["policy_goal"] >= 50,
            "policy_far_at_least_100": counts["policy_far"] >= 100,
            "stop_auc_at_least_0_80": metrics["stop_auc"] >= .80,
            "instruction_swap_accuracy_at_least_0_80":
                metrics["instruction_swap_accuracy"] >= .80,
            "pooled_fpr_at_most_0_10": metrics["pooled_fpr"] <= .10,
            "pooled_recall_at_least_0_50": metrics["pooled_recall"] >= .50,
            "wrong_instruction_fpr_at_most_0_15":
                metrics["wrong_instruction_fpr"] <= .15,
            "policy_far_fpr_at_most_0_10": metrics["policy_far_fpr"] <= .10,
            "policy_goal_recall_at_least_0_50":
                metrics["policy_goal_recall"] >= .50}
    result = {"schema": "policy_stop_hardneg_reused_audit_v1",
              "checkpoint_sha256": digest(args.checkpoint),
              "development_sha256": digest(args.development),
              "selected_step": checkpoint["selected_step"],
              "threshold": threshold, "audit_scenes": len(data["scenes"]),
              "counts": counts, "metrics": metrics,
              "predeclared_gate": gate,
              "eligible_for_group4_stop_pilot": all(gate.values()),
              "elapsed_seconds": time.time() - started,
              "interpretation": (
                  "Reused R2R-train audit scenes previously inspected during method "
                  "development; exploratory transfer screen only. No independent "
                  "semantic truth or navigation gain measured."
              )}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
