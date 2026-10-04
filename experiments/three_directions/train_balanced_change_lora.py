"""Balanced three-class, instruction-conditioned visual change probe.

The start/before/after RGB, executed action, and instruction are available
observations. Geodesic distances are labels only. Fit uses the audited
diversity-first train-scene extension; development is the unchanged old
scene-disjoint split. No audit or val-unseen source is loaded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part, signed_pairs
from train_action_memory_progress_lora import build_inputs, index_both
from train_policy_progress_lora import (choose, fixed_subset, gate,
                                        index_experts, indexed, pair_indices)


EXPANDED_SHA = "bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37"
OLD_SHA = "aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681"
CLASSES = ("forward", "backward", "stationary")


def load_expanded_fit(common: tuple[Path, ...], expanded_manifest: Path,
                      expanded_root: Path, fit_audit: Path) -> tuple[dict, dict]:
    fit = load_part("fit", *common)
    if digest(common[4]) != OLD_SHA or digest(expanded_manifest) != EXPANDED_SHA:
        raise ValueError("old or expanded policy source changed")
    old = json.loads(common[4].read_text())
    new = json.loads(expanded_manifest.read_text())
    audit = json.loads(fit_audit.read_text())
    if new["schema"] != "policy_process_train_manifest_v1" or \
            new["targets"] != {"fit": 1024, "development": 320, "audit": 320} or \
            new["selected"]["fit"][:768] != old["selected"]["fit"] or \
            new["selected"]["development"] != old["selected"]["development"] or \
            new["selected"]["audit"] != old["selected"]["audit"] or \
            audit["schema"] != "diversity_policy_fit_replay_audit_v1" or \
            audit["new_manifest_sha256"] != EXPANDED_SHA or \
            not audit["extra_sample_gate"]["passed"]:
        raise ValueError("fit expansion audit or scene isolation invalid")
    extra = []
    root = expanded_root / "fit"
    for plan in new["selected"]["fit"][768:]:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        record = json.loads((root / "records" / f"{rid}.json").read_text())
        if record["manifest_sha256"] != EXPANDED_SHA or \
                record["record_id"] != rid or \
                record["scene_id"] != plan["scene_id"] or \
                record["terminal_mode"] != plan["terminal_mode"] or \
                record["scene_id"] not in fit["scenes"]:
            raise ValueError(f"expanded fit record mismatch {rid}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        extra.append({"record": record, "root": root,
                      "distances": distances, "pairs": signed_pairs(distances)})
    if len(fit["policy"]) != 768 or len(extra) != 256:
        raise ValueError("expanded fit count mismatch")
    fit["policy"].extend(extra)
    return fit, audit


def logits_for(processor, model, head, item: dict,
               before: int, after: int,
               instruction: str | None = None) -> torch.Tensor:
    inputs = build_inputs(processor, item, before, after, instruction).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    values = head(output.logits[0, -1].float())
    if values.shape != (3,):
        raise ValueError("three-class head shape changed")
    return values


def confidence(values: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(values.float(), dim=-1)
    return probabilities[0] - probabilities[1]


def local_loss(processor, model, head, item: dict,
               after: int, kind: str) -> torch.Tensor:
    meters = item["distances"][after - 1] - item["distances"][after]
    if kind == "forward" and meters < 1.0 or \
            kind == "backward" and meters > -1.0 or \
            kind == "stationary" and abs(meters) >= .1:
        raise ValueError("three-class label/turn mismatch")
    values = logits_for(processor, model, head, item, after - 1, after)
    target = torch.as_tensor([CLASSES.index(kind)], device=values.device)
    loss = F.cross_entropy(values[None, :], target)
    if kind == "stationary":
        loss = loss + .2 * confidence(values).square()
    return loss


def within_loss(processor, model, head, item: dict,
                rng: random.Random) -> torch.Tensor:
    pairs = pair_indices(item)
    if not pairs["forward"] or not pairs["backward"]:
        raise ValueError("within-trajectory contrast unavailable")
    f = rng.choice(pairs["forward"])
    b = rng.choice(pairs["backward"])
    positive = logits_for(processor, model, head, item, f - 1, f)
    negative = logits_for(processor, model, head, item, b - 1, b)
    forward_margin = positive[0] - positive[1]
    backward_margin = negative[0] - negative[1]
    return F.softplus(.4 - (forward_margin - backward_margin))


def instruction_loss(processor, model, head, item: dict) -> torch.Tensor:
    if not item["safe_swap"]:
        raise ValueError("unsafe wrong-goal instruction")
    record = item["record"]
    end = len(record["turns"])
    correct = logits_for(processor, model, head, item, 0, end,
                         record["instruction"])
    wrong = logits_for(processor, model, head, item, 0, end,
                       record["wrong_instruction"])
    correct_margin = correct[0] - correct[2]
    wrong_margin = wrong[0] - wrong[2]
    return F.softplus(.3 - (correct_margin - wrong_margin)) + \
        .25 * F.softplus(.2 - correct_margin)


def evaluate(processor, model, head, data: dict, *, small: bool) -> dict:
    model.eval()
    head.eval()
    policies = data["policy"]
    experts = [item for item in data["expert"] if item["safe_swap"]]
    if small:
        policies = fixed_subset(policies, 96, "action-memory-policy-dev")
        experts = fixed_subset(experts, 48, "action-memory-expert-dev")
    hit = {kind: 0 for kind in ("forward", "backward")}
    total = {kind: 0 for kind in hit}
    by_scene = {kind: defaultdict(list) for kind in hit}
    episodes = {kind: set() for kind in CLASSES}
    stationary, forward = [], []
    instruction_hits = 0
    with torch.inference_mode():
        for item in policies:
            record = item["record"]
            for kind, afters in pair_indices(item).items():
                for after in afters:
                    value = float(confidence(logits_for(
                        processor, model, head, item, after - 1, after)))
                    if not math.isfinite(value):
                        raise ValueError("nonfinite development score")
                    episodes[kind].add(str(record["episode_id"]))
                    if kind == "stationary":
                        stationary.append(value)
                    elif kind == "forward":
                        correct = value > 0
                        hit[kind] += correct
                        total[kind] += 1
                        by_scene[kind][record["scene_id"]].append(correct)
                        forward.append(value)
                    else:
                        correct = value < 0
                        hit[kind] += correct
                        total[kind] += 1
                        by_scene[kind][record["scene_id"]].append(correct)
        for item in experts:
            record = item["record"]
            end = len(record["turns"])
            correct = torch.softmax(logits_for(
                processor, model, head, item, 0, end,
                record["instruction"]), -1)[0]
            wrong = torch.softmax(logits_for(
                processor, model, head, item, 0, end,
                record["wrong_instruction"]), -1)[0]
            instruction_hits += float(correct) > float(wrong)
    if not all(total.values()) or not stationary or not forward or not experts:
        raise ValueError("missing development class")
    ordered = sorted(stationary)
    threshold = max(0.0, ordered[min(len(ordered) - 1,
                                     math.ceil(.9 * len(ordered)) - 1)])
    accuracy = {kind: hit[kind] / total[kind] for kind in hit}
    macros = {kind: sum(sum(values) / len(values) for values in
                        by_scene[kind].values()) / len(by_scene[kind])
              for kind in hit}
    return {
        "policy_trajectories": len(policies),
        "expert_swap_trajectories": len(experts),
        "pair_counts": {**total, "stationary": len(stationary)},
        "unique_episodes_by_class": {kind: len(ids)
                                     for kind, ids in episodes.items()},
        "accuracy": accuracy, "scene_macro_accuracy": macros,
        "balanced_direction_accuracy": .5 * sum(accuracy.values()),
        "instruction_gain_preference": instruction_hits / len(experts),
        "instruction_pairs": len(experts),
        "stationary_threshold_selected_on_development": threshold,
        "stationary_false_positive_rate": sum(x > threshold for x in stationary) / len(ordered),
        "forward_recall_at_stationary_threshold": sum(x > threshold for x in forward) / len(forward),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "expanded-manifest", "expanded-root",
                 "expanded-audit", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed, steps, accumulation = 11, (5 if args.smoke else 1500), 5
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit, expanded_audit = load_expanded_fit(
        common, args.expanded_manifest, args.expanded_root,
        args.expanded_audit)
    dev = None if args.smoke else load_part("development", *common)
    safe_experts = [item for item in fit["expert"] if item["safe_swap"]]
    expert_index = index_experts(safe_experts)
    by_class = {kind: indexed(fit["policy"], kind) for kind in CLASSES}
    both = index_both(fit["policy"])
    processor, model, old_head, trainable = load_model(args.model)
    hidden_size = old_head.net[0].normalized_shape[0]
    del old_head
    head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 128),
                         nn.GELU(), nn.Linear(128, 3)).cuda()
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 7e-5},
        {"params": head.parameters(), "lr": 2e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    best, history = None, []
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        kind = ("forward", "backward", "stationary", "instruction",
                "within_trajectory")[(step - 1) % 5]
        if kind == "instruction":
            loss = instruction_loss(processor, model, head,
                                    choose(expert_index, rng))
        elif kind == "within_trajectory":
            loss = within_loss(processor, model, head,
                               choose(both, rng), rng)
        else:
            item = choose(by_class[kind], rng)
            after = rng.choice(pair_indices(item)[kind])
            loss = local_loss(processor, model, head, item, after, kind)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite {kind} loss at step {step}")
        (loss / accumulation).backward()
        if step % accumulation == 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(gradient) or float(gradient) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "kind": kind,
                              "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}), flush=True)
        if not args.smoke and step in (250, 500, 1000, 1500):
            metrics = evaluate(processor, model, head, dev, small=True)
            decision = gate(metrics, preliminary=True)
            history.append({"step": step, "development_small": metrics,
                            "preliminary_gate": decision})
            print(json.dumps(history[-1]), flush=True)
            quality = (sum(decision.values()),
                       metrics["balanced_direction_accuracy"],
                       metrics["instruction_gain_preference"], -step)
            if best is None or quality > best[0]:
                best = (quality, step,
                        {k: v.detach().cpu().clone() for k, v in
                         get_peft_model_state_dict(model).items()},
                        copy.deepcopy({k: v.detach().cpu() for k, v in
                                       head.state_dict().items()}), metrics)
    if args.smoke:
        print(json.dumps({"smoke_updates": 1,
                          "lora_parameters": trainable,
                          "fit_policy_trajectories": len(fit["policy"]),
                          "fit_regression_turns": expanded_audit["combined_fit"]["counts"]["regression_turns_1m"],
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no selected checkpoint")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    full_dev = (evaluate(processor, model, head, dev, small=False)
                if all(gate(best[4], preliminary=True).values()) else None)
    report = {
        "schema": "balanced_visual_change_lora_development_v1",
        "interpretation": "Train-scene development only; no audit or navigation result.",
        "seed": seed, "microsteps": steps,
        "gradient_accumulation": accumulation,
        "lora_parameters": trainable, "selected_step": best[1],
        "source_sha256": {"scene_split": digest(args.scene_split),
                          "expert_manifest": digest(args.expert_manifest),
                          "old_policy_manifest": digest(args.policy_manifest),
                          "expanded_policy_manifest": digest(args.expanded_manifest),
                          "expanded_fit_audit": digest(args.expanded_audit),
                          "model_config": digest(args.model / "config.json")},
        "fit_policy_trajectories": len(fit["policy"]),
        "fit_safe_expert_trajectories": len(safe_experts),
        "fit_regression_turns": expanded_audit["combined_fit"]["counts"]["regression_turns_1m"],
        "selected_small_development": best[4],
        "small_development_gate": gate(best[4], preliminary=True),
        "full_development": full_dev,
        "full_development_gate": gate(full_dev, preliminary=False)
                                 if full_dev is not None else None,
        "history": history, "elapsed_seconds": time.time() - started,
    }
    torch.save({"schema": "balanced_visual_change_lora_v1",
                "adapter": best[2], "head": best[3],
                "selected_step": best[1],
                "source_sha256": report["source_sha256"]},
               args.output / "adapter_head.pt")
    (args.output / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1],
                      "small_gate": report["small_development_gate"],
                      "full_gate": report["full_development_gate"]}),
          flush=True)


if __name__ == "__main__":
    main()
