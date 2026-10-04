"""Fit an observation-only, instruction-grounded local progress adapter.

This is an offline representation screen on R2R-train scenes. Geodesic
distance selects/labels movement pairs, but never enters a model prompt.
The old audit split and val-unseen are absent from the CLI. The prompt
uses the online policy's stripped-system preprocessing path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part, score


def indexed(items: list[dict], kind: str) -> dict:
    by_scene = defaultdict(lambda: defaultdict(list))
    for item in items:
        if not pair_indices(item)[kind]:
            continue
        record = item["record"]
        by_scene[record["scene_id"]][str(record["episode_id"])].append(item)
    if not by_scene:
        raise ValueError(f"no {kind} examples")
    return by_scene


def choose(index: dict, rng: random.Random) -> dict:
    scene = rng.choice(sorted(index))
    episode = rng.choice(sorted(index[scene]))
    return rng.choice(index[scene][episode])


def index_experts(items: list[dict]) -> dict:
    by_scene = defaultdict(lambda: defaultdict(list))
    for item in items:
        record = item["record"]
        by_scene[record["scene_id"]][str(record["episode_id"])].append(item)
    if not by_scene:
        raise ValueError("no safe expert instruction swaps")
    return by_scene


def pair_indices(item: dict) -> dict[str, list[int]]:
    distances = item["distances"]
    result = {"forward": [], "backward": [], "stationary": []}
    for after in range(1, len(distances)):
        delta = distances[after - 1] - distances[after]
        if delta >= 1.0:
            result["forward"].append(after)
        elif delta <= -1.0:
            result["backward"].append(after)
        elif abs(delta) < .1:
            result["stationary"].append(after)
    return result


def progress(processor, model, head, item: dict, count: int,
             instruction: str | None = None) -> torch.Tensor:
    _, value = score(processor, model, head, item, count, instruction,
                     include_system=False)
    return value


def local_loss(processor, model, head, item: dict, after: int,
               kind: str) -> torch.Tensor:
    before = progress(processor, model, head, item, after - 1)
    later = progress(processor, model, head, item, after)
    difference = later - before
    meters = item["distances"][after - 1] - item["distances"][after]
    if kind == "stationary":
        if abs(meters) >= .1:
            raise ValueError("stationary label mismatch")
        return F.smooth_l1_loss(difference, torch.zeros_like(difference),
                                beta=.1) + F.relu(difference - .05)
    direction = 1.0 if kind == "forward" else -1.0
    if meters * direction < 1.0:
        raise ValueError("signed label mismatch")
    bounded_target = math.tanh(meters / 3.0)
    target = torch.as_tensor(bounded_target, dtype=difference.dtype,
                             device=difference.device)
    return (F.softplus(.2 - direction * difference) +
            .5 * F.smooth_l1_loss(difference, target, beta=.15))


def instruction_loss(processor, model, head, item: dict) -> torch.Tensor:
    record = item["record"]
    if not item["safe_swap"]:
        raise ValueError("unsafe instruction swap")
    wrong = record["wrong_instruction"].strip()
    original = record["instruction"].strip()
    if not wrong or wrong == original:
        raise ValueError("missing distinct instruction")
    end = len(record["turns"])
    correct_gain = (progress(processor, model, head, item, end) -
                    progress(processor, model, head, item, 0))
    wrong_gain = (progress(processor, model, head, item, end, wrong) -
                  progress(processor, model, head, item, 0, wrong))
    return (F.softplus(.2 - correct_gain) +
            F.softplus(.2 - (correct_gain - wrong_gain)))


def fixed_subset(items: list[dict], limit: int, salt: str) -> list[dict]:
    def key(item: dict) -> str:
        record = item["record"]
        identity = record.get("record_id", str(record["episode_id"]))
        return hashlib.sha256(f"{salt}|{identity}".encode()).hexdigest()
    return sorted(items, key=key)[:limit]


def evaluate(processor, model, head, data: dict,
             *, small: bool) -> dict:
    model.eval()
    head.eval()
    policies = data["policy"]
    experts = [item for item in data["expert"] if item["safe_swap"]]
    if small:
        policies = fixed_subset(policies, 96, "progress-policy-dev")
        experts = fixed_subset(experts, 48, "progress-expert-dev")
    hit = {kind: 0 for kind in ("forward", "backward")}
    total = {kind: 0 for kind in hit}
    scene_hit = {kind: defaultdict(list) for kind in hit}
    stationary_values = []
    forward_values = []
    swap_correct = 0
    episode_classes = {kind: set() for kind in
                       ("forward", "backward", "stationary")}
    with torch.inference_mode():
        for item in policies:
            record = item["record"]
            pairs = pair_indices(item)
            needed = {j for afters in pairs.values()
                      for after in afters for j in (after - 1, after)}
            if not needed:
                continue
            values = {j: float(progress(processor, model, head, item, j))
                      for j in sorted(needed)}
            scene = record["scene_id"]
            eid = str(record["episode_id"])
            for kind, afters in pairs.items():
                for after in afters:
                    delta = values[after] - values[after - 1]
                    episode_classes[kind].add(eid)
                    if kind == "stationary":
                        stationary_values.append(delta)
                    elif kind == "forward":
                        correct = delta > 0
                        hit[kind] += correct
                        total[kind] += 1
                        scene_hit[kind][scene].append(correct)
                        forward_values.append(delta)
                    else:
                        correct = delta < 0
                        hit[kind] += correct
                        total[kind] += 1
                        scene_hit[kind][scene].append(correct)
        for item in experts:
            record = item["record"]
            end = len(record["turns"])
            wrong = record["wrong_instruction"].strip()
            correct_gain = (float(progress(processor, model, head, item, end)) -
                            float(progress(processor, model, head, item, 0)))
            wrong_gain = (float(progress(processor, model, head, item, end, wrong)) -
                          float(progress(processor, model, head, item, 0, wrong)))
            swap_correct += correct_gain > wrong_gain
    if not all(total.values()) or not stationary_values or not experts:
        raise ValueError("development class missing")
    if not all(math.isfinite(value) for value in
               stationary_values + forward_values):
        raise ValueError("nonfinite development scores")
    ordered = sorted(stationary_values)
    threshold = max(0.0, ordered[min(len(ordered) - 1,
                                     math.ceil(.9 * len(ordered)) - 1)])
    stationary_fpr = sum(value > threshold for value in stationary_values) / len(ordered)
    forward_recall = sum(value > threshold for value in forward_values) / len(forward_values)
    accuracy = {kind: hit[kind] / total[kind] for kind in hit}
    macros = {kind: sum(sum(values) / len(values) for values in scene_hit[kind].values()) /
                    len(scene_hit[kind]) for kind in hit}
    return {"policy_trajectories": len(policies),
            "expert_swap_trajectories": len(experts),
            "pair_counts": {**total, "stationary": len(stationary_values)},
            "unique_episodes_by_class": {kind: len(ids) for kind, ids in
                                         episode_classes.items()},
            "accuracy": accuracy,
            "scene_macro_accuracy": macros,
            "balanced_direction_accuracy": .5 * sum(accuracy.values()),
            "instruction_gain_preference": swap_correct / len(experts),
            "instruction_pairs": len(experts),
            "stationary_threshold_selected_on_development": threshold,
            "stationary_false_positive_rate": stationary_fpr,
            "forward_recall_at_stationary_threshold": forward_recall}


def gate(metrics: dict, preliminary: bool) -> dict[str, bool]:
    balance = .70 if preliminary else .75
    class_min = .60 if preliminary else .65
    grounding = .70 if preliminary else .75
    recall = .40 if preliminary else .50
    return {
        "balanced_direction": metrics["balanced_direction_accuracy"] >= balance,
        "forward": metrics["accuracy"]["forward"] >= class_min,
        "backward": metrics["accuracy"]["backward"] >= class_min,
        "instruction_grounding": metrics["instruction_gain_preference"] >= grounding,
        "stationary_fpr": metrics["stationary_false_positive_rate"] <= .10 + 1e-9,
        "forward_recall": metrics["forward_recall_at_stationary_threshold"] >= recall,
    }


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed, steps, accumulation = 11, (4 if args.smoke else 256), 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit = load_part("fit", *common)
    dev = None if args.smoke else load_part("development", *common)
    safe_experts = [item for item in fit["expert"] if item["safe_swap"]]
    if not safe_experts:
        raise ValueError("no safe expert swaps")
    expert_index = index_experts(safe_experts)
    by_class = {kind: indexed(fit["policy"], kind) for kind in
                ("forward", "backward", "stationary")}
    processor, model, head, trainable = load_model(args.model)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 7e-5},
        {"params": head.parameters(), "lr": 2e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    best = None
    history = []
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        kind = ("forward", "backward", "stationary", "instruction")[(step - 1) % 4]
        if kind == "instruction":
            item = choose(expert_index, rng)
            loss = instruction_loss(processor, model, head, item)
        else:
            item = choose(by_class[kind], rng)
            after = rng.choice(pair_indices(item)[kind])
            loss = local_loss(processor, model, head, item, after, kind)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite {kind} loss at microstep {step}")
        (loss / accumulation).backward()
        if step % accumulation == 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(gradient) or float(gradient) <= 0:
                raise ValueError(f"invalid gradient at microstep {step}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 16 == 0:
            print(json.dumps({"step": step, "kind": kind,
                              "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}), flush=True)
        if not args.smoke and step in (128, 256):
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
        print(json.dumps({"smoke_updates": steps // accumulation,
                          "lora_parameters": trainable,
                          "fit_policy": len(fit["policy"]),
                          "fit_safe_expert": len(safe_experts),
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no development-selected checkpoint")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    full_dev = (evaluate(processor, model, head, dev, small=False)
                if all(gate(best[4], preliminary=True).values()) else None)
    report = {
        "schema": "policy_progress_instruction_lora_development_v1",
        "seed": seed, "microsteps": steps, "gradient_accumulation": accumulation,
        "lora_parameters": trainable,
        "selected_step": best[1],
        "source_sha256": {"scene_split": digest(args.scene_split),
                          "expert_manifest": digest(args.expert_manifest),
                          "policy_manifest": digest(args.policy_manifest),
                          "model_config": digest(args.model / "config.json")},
        "prompt": "policy-format history, stripped system block, PNG processor path",
        "fit_policy_trajectories": len(fit["policy"]),
        "fit_safe_expert_trajectories": len(safe_experts),
        "selected_small_development": best[4],
        "small_development_gate": gate(best[4], preliminary=True),
        "full_development": full_dev,
        "full_development_gate": gate(full_dev, preliminary=False)
                                 if full_dev is not None else None,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "interpretation": "R2R-train development screen only; no audit or navigation result.",
    }
    torch.save({"schema": "policy_progress_instruction_lora_v1",
                "adapter": best[2], "head": best[3],
                "selected_step": best[1],
                "source_sha256": report["source_sha256"]},
               args.output / "adapter_head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1],
                      "small_gate": report["small_development_gate"],
                      "full_gate": report["full_development_gate"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
