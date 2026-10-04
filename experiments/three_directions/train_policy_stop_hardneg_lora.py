"""Train a policy-prompt STOP reward with explicit policy failure negatives.

This is a train-scene representation screen, not a navigation result. The
original history-grounding run weakly supervised STOP on policy steps that
also had a >=1 m progress change. Here every other microstep instead draws
an episode-balanced near-goal or far-goal terminal policy history. The
navigation policy's multi-turn prompt has no system block, so neither does
this reward model. Only fit scenes update weights; development scenes select
the checkpoint and STOP threshold. The audit partition is never loaded.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import (digest, load_model, load_part,
                                    sampling_index, score)
from stop_history_head import auc
from train_history_grounding_lora import select_subset


def policy_index(items: list[dict]) -> dict:
    result = {"goal": defaultdict(lambda: defaultdict(list)),
              "far": defaultdict(lambda: defaultdict(list))}
    for item in items:
        distance = item["distances"][-1]
        category = "goal" if distance <= 3.0 else (
            "far" if distance >= 3.5 else None)
        if category is not None:
            record = item["record"]
            result[category][record["scene_id"]][
                str(record["episode_id"])].append(item)
    if not all(result.values()):
        raise ValueError("missing policy terminal class")
    return result


def sample_terminal(index: dict, label: str, rng: random.Random) -> dict:
    scenes = index[label]
    episodes = rng.choice(list(scenes[rng.choice(list(scenes))].values()))
    return rng.choice(episodes)


def train_expert(processor, model, head, index: dict,
                 rng: random.Random) -> torch.Tensor:
    items = index[rng.choice(list(index))]
    item = rng.choice(items)
    record = item["record"]
    end = len(record["turns"])
    positive, _ = score(processor, model, head, item, end,
                        include_system=False)
    if item["safe_swap"] and rng.random() < .75:
        negative, _ = score(processor, model, head, item, end,
                            record["wrong_instruction"],
                            include_system=False)
    else:
        negative, _ = score(processor, model, head, item, 0,
                            include_system=False)
    return (F.softplus(-positive) + 1.5 * F.softplus(negative) +
            .5 * F.softplus(1.0 - (positive - negative)))


def train_policy(processor, model, head, index: dict,
                 rng: random.Random) -> torch.Tensor:
    label = rng.choice(("goal", "far"))
    item = sample_terminal(index, label, rng)
    logit, _ = score(processor, model, head, item,
                     len(item["record"]["turns"]), include_system=False)
    return F.softplus(-logit if label == "goal" else logit)


def choose_conservative_threshold(positive: list[float],
                                  negative: list[float]) -> dict:
    """Maximize dev recall subject to a predeclared 5% dev FPR cap."""
    if not positive or not negative:
        raise ValueError("empty STOP development class")
    candidates = sorted(set(positive + negative), reverse=True)
    feasible = []
    for threshold in [max(candidates) + 1.0] + candidates:
        false_rate = sum(x >= threshold for x in negative) / len(negative)
        if false_rate <= .05:
            hit_rate = sum(x >= threshold for x in positive) / len(positive)
            feasible.append((hit_rate, -false_rate, threshold))
    recall, neg_fpr, threshold = max(feasible)
    return {"threshold": threshold, "development_fpr": -neg_fpr,
            "development_recall": recall, "target_fpr": .05}


def evaluate(processor, model, head, data: dict, small: bool) -> dict:
    model.eval()
    head.eval()
    experts = (select_subset(data["expert"], 64, "hardneg-dev-expert")
               if small else data["expert"])
    policies = (select_subset(data["policy"], 128, "hardneg-dev-policy")
                if small else data["policy"])
    classes = {name: [] for name in
               ("expert_goal", "expert_start", "expert_wrong",
                "policy_goal", "policy_far")}
    swaps = 0
    with torch.inference_mode():
        for item in experts:
            end = len(item["record"]["turns"])
            good, _ = score(processor, model, head, item, end,
                            include_system=False)
            start, _ = score(processor, model, head, item, 0,
                             include_system=False)
            classes["expert_goal"].append(float(good))
            classes["expert_start"].append(float(start))
            if item["safe_swap"]:
                wrong, _ = score(processor, model, head, item, end,
                                 item["record"]["wrong_instruction"],
                                 include_system=False)
                classes["expert_wrong"].append(float(wrong))
                swaps += int(float(good) > float(wrong))
        for item in policies:
            distance = item["distances"][-1]
            if 3.0 < distance < 3.5:
                continue
            logit, _ = score(processor, model, head, item,
                             len(item["record"]["turns"]),
                             include_system=False)
            classes["policy_goal" if distance <= 3.0 else
                    "policy_far"].append(float(logit))
    if any(not values for values in classes.values()):
        raise ValueError("incomplete development STOP class")
    positive = classes["expert_goal"] + classes["policy_goal"]
    negative = (classes["expert_start"] + classes["expert_wrong"] +
                classes["policy_far"])
    threshold = choose_conservative_threshold(positive, negative)
    value = threshold["threshold"]
    result = {"stop_auc": auc(torch.tensor(positive),
                               torch.tensor(negative)),
              "instruction_swap_accuracy": swaps / len(classes["expert_wrong"]),
              "wrong_instruction_fpr": sum(x >= value for x in classes[
                  "expert_wrong"]) / len(classes["expert_wrong"]),
              "policy_far_fpr": sum(x >= value for x in classes[
                  "policy_far"]) / len(classes["policy_far"]),
              "policy_goal_recall": sum(x >= value for x in classes[
                  "policy_goal"]) / len(classes["policy_goal"]),
              "counts": {key: len(values) for key, values in classes.items()},
              "threshold": threshold}
    return result


def development_gate(metrics: dict) -> dict:
    threshold = metrics["threshold"]
    return {"stop_auc_at_least_0_80": metrics["stop_auc"] >= .80,
            "instruction_swap_accuracy_at_least_0_80":
                metrics["instruction_swap_accuracy"] >= .80,
            "pooled_fpr_at_most_0_05":
                threshold["development_fpr"] <= .05,
            "pooled_recall_at_least_0_55":
                threshold["development_recall"] >= .55,
            "wrong_instruction_fpr_at_most_0_12":
                metrics["wrong_instruction_fpr"] <= .12,
            "policy_far_fpr_at_most_0_08":
                metrics["policy_far_fpr"] <= .08,
            "policy_goal_recall_at_least_0_50":
                metrics["policy_goal_recall"] >= .50}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed, steps, grad_accum, eval_every = 11, (4 if args.smoke else 512), 4, 128
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    fit = load_part("fit", args.scene_split, args.expert_manifest,
                    args.expert_labels, args.expert_root,
                    args.policy_manifest, args.policy_root, args.policy_audit)
    dev = None if args.smoke else load_part(
        "development", args.scene_split, args.expert_manifest,
        args.expert_labels, args.expert_root,
        args.policy_manifest, args.policy_root, args.policy_audit)
    processor, model, head, trainable = load_model(args.model)
    expert_index = sampling_index(fit["expert"])
    terminal_index = policy_index(fit["policy"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": 1e-4},
        {"params": head.parameters(), "lr": 3e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    best, history = None, []
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        loss = (train_expert(processor, model, head, expert_index, rng)
                if step % 2 else
                train_policy(processor, model, head, terminal_index, rng))
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        (loss / grad_accum).backward()
        if step % grad_accum == 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(gradient) or float(gradient) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}),
                  flush=True)
        if not args.smoke and step % eval_every == 0:
            metrics = evaluate(processor, model, head, dev, small=True)
            history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
            threshold = metrics["threshold"]
            quality = (min(metrics["stop_auc"] - .80,
                           metrics["instruction_swap_accuracy"] - .80,
                           threshold["development_recall"] - .55,
                           .12 - metrics["wrong_instruction_fpr"],
                           .08 - metrics["policy_far_fpr"]),
                       metrics["stop_auc"], -step)
            if best is None or quality > best[0]:
                best = (quality, step,
                        {k: v.detach().cpu().clone() for k, v in
                         get_peft_model_state_dict(model).items()},
                        copy.deepcopy({k: v.detach().cpu() for k, v in
                                       head.state_dict().items()}))
                tmp = args.output / "interim_selected.tmp"
                torch.save({"selected_step": step, "adapter": best[2],
                            "head": best[3]}, tmp)
                os.replace(tmp, args.output / "interim_selected.pt")
    if args.smoke:
        print(json.dumps({"smoke_updates": steps // grad_accum,
                          "lora_parameters": trainable,
                          "fit_expert": len(fit["expert"]),
                          "fit_policy": len(fit["policy"]),
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no checkpoint selected")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    metrics = evaluate(processor, model, head, dev, small=False)
    gate = development_gate(metrics)
    sources = {"scene_split": fit["scene_split_sha256"],
               "expert_manifest": fit["expert_manifest_sha256"],
               "policy_manifest": fit["policy_manifest_sha256"]}
    checkpoint = {"schema": "policy_stop_hardneg_lora_seed11_v1",
                  "adapter": best[2], "head": best[3],
                  "selected_step": best[1], "source_sha256": sources,
                  "model_config_sha256": digest(args.model / "config.json"),
                  "development_stop_threshold": metrics["threshold"]["threshold"],
                  "prompt_mode": "policy_multiturn_without_system"}
    torch.save(checkpoint, args.output / "adapter_head.pt")
    report = {"schema": "policy_stop_hardneg_lora_development_v1",
              "selected_step": best[1], "training_microsteps": steps,
              "gradient_accumulation": grad_accum,
              "lora_parameters": trainable, "fit_expert": len(fit["expert"]),
              "fit_policy": len(fit["policy"]),
              "development_expert": len(dev["expert"]),
              "development_policy": len(dev["policy"]),
              "source_sha256": sources, "development": metrics,
              "predeclared_gate": gate,
              "eligible_for_locked_audit": all(gate.values()),
              "history": history, "elapsed_seconds": time.time() - started,
              "interpretation": "Train-scene representation screen only; no navigation gain."}
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1], "development": metrics,
                      "predeclared_gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
