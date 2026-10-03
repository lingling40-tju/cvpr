"""Fit a fresh navigation-SFT LoRA on instruction grounding and real regressions.

Uses only fit-scene gradients. Development scenes select the checkpoint
and STOP threshold. The audit split is deliberately absent from this CLI.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import load_model, load_part, sampling_index, score
from stop_history_head import auc, choose_threshold


def select_subset(rows: list[dict], size: int, salt: str) -> list[dict]:
    def key(item: dict) -> str:
        record = item["record"]
        identity = record.get("record_id", record["episode_id"])
        return hashlib.sha256(f"{salt}|{identity}".encode()).hexdigest()
    return sorted(rows, key=key)[:size]


def evaluate(processor, model, head, data: dict,
             small: bool) -> dict:
    model.eval()
    head.eval()
    experts = data["expert"]
    policies = data["policy"]
    if small:
        experts = select_subset(experts, 64, "stop-lora-dev-expert")
        policies = select_subset(policies, 128, "stop-lora-dev-policy")
    positives, negatives = [], []
    swap_correct, swap_total = 0, 0
    forward_correct, backward_correct = 0, 0
    forward_total, backward_total = 0, 0
    begun = time.time()
    with torch.inference_mode():
        for item in experts:
            record = item["record"]
            end = len(record["turns"])
            positive, _ = score(processor, model, head, item, end)
            negative, _ = score(processor, model, head, item, 0)
            positives.append(float(positive))
            negatives.append(float(negative))
            if item["safe_swap"]:
                wrong, _ = score(processor, model, head, item, end,
                                 record["wrong_instruction"])
                negatives.append(float(wrong))
                swap_correct += bool(positive > wrong)
                swap_total += 1
        for item in policies:
            record = item["record"]
            distances = item["distances"]
            pairs = item["pairs"]
            afters = set(pairs["backward"])
            afters.update(pairs["forward"][:2])
            needed = {len(record["turns"])}
            for after in afters:
                needed.add(after - 1)
                needed.add(after)
            cached = {index: score(processor, model, head, item, index)
                      for index in sorted(needed)}
            terminal = float(cached[len(record["turns"])][0])
            if distances[-1] <= 3.0:
                positives.append(terminal)
            elif distances[-1] >= 3.5:
                negatives.append(terminal)
            for after in afters:
                before_value = cached[after - 1][1]
                after_value = cached[after][1]
                if after in pairs["backward"]:
                    backward_correct += bool(after_value < before_value)
                    backward_total += 1
                else:
                    forward_correct += bool(after_value > before_value)
                    forward_total += 1
    if not positives or not negatives or not swap_total or \
            not forward_total or not backward_total:
        raise ValueError("development metric class missing")
    positive_t = torch.tensor(positives)
    negative_t = torch.tensor(negatives)
    return {"stop_auc": auc(positive_t, negative_t),
            "stop_positive": len(positives), "stop_negative": len(negatives),
            "instruction_swap_accuracy": swap_correct / swap_total,
            "instruction_swap_pairs": swap_total,
            "forward_accuracy": forward_correct / forward_total,
            "forward_pairs": forward_total,
            "regression_accuracy": backward_correct / backward_total,
            "regression_pairs": backward_total,
            "balanced_progress_accuracy": .5 * (
                forward_correct / forward_total + backward_correct / backward_total),
            "positive_logits": positives, "negative_logits": negatives,
            "elapsed_seconds": time.time() - begun}


def sample_scene(index: dict, rng: random.Random) -> dict:
    scene = rng.choice(list(index))
    return rng.choice(index[scene])


def train_expert(processor, model, head, index: dict,
                 rng: random.Random) -> torch.Tensor:
    item = sample_scene(index, rng)
    record = item["record"]
    end = len(record["turns"])
    positive, _ = score(processor, model, head, item, end)
    if item["safe_swap"] and rng.random() < .75:
        negative, _ = score(processor, model, head, item, end,
                            record["wrong_instruction"])
    else:
        negative, _ = score(processor, model, head, item, 0)
    return (F.softplus(-positive) + F.softplus(negative) +
            .5 * F.softplus(1.0 - (positive - negative)))


def train_policy(processor, model, head, indices: dict,
                 rng: random.Random) -> torch.Tensor:
    category = rng.choice(("forward", "backward"))
    item = sample_scene(indices[category], rng)
    after = rng.choice(item["pairs"][category])
    before_stop, before_progress = score(processor, model, head,
                                         item, after - 1)
    after_stop, after_progress = score(processor, model, head,
                                      item, after)
    direction = 1.0 if category == "forward" else -1.0
    loss = F.softplus(-direction * (after_progress - before_progress))
    for index, logit in ((after - 1, before_stop), (after, after_stop)):
        distance = item["distances"][index]
        if distance <= 3.0:
            loss = loss + .25 * F.softplus(-logit)
        elif distance >= 3.5:
            loss = loss + .25 * F.softplus(logit)
    return loss


def stripped(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items()
            if key not in ("positive_logits", "negative_logits")}


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
    seed, max_steps, grad_accum, eval_every = 11, (4 if args.smoke else 512), 4, 128
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit = load_part("fit", *common)
    dev = None if args.smoke else load_part("development", *common)
    processor, model, head, trainable = load_model(args.model)
    expert_index = sampling_index(fit["expert"])
    policy_indices = {kind: sampling_index(fit["policy"], kind)
                      for kind in ("forward", "backward")}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": head.parameters(), "lr": 3e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    best = None
    history = []
    started = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        head.train()
        if step % 2:
            loss = train_expert(processor, model, head, expert_index, rng)
        else:
            loss = train_policy(processor, model, head, policy_indices, rng)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training loss at step {step}")
        (loss / grad_accum).backward()
        if step % grad_accum == 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                list(p for p in model.parameters() if p.requires_grad) +
                list(head.parameters()), 1.0)
            if not torch.isfinite(gradient) or float(gradient) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}), flush=True)
        if not args.smoke and step % eval_every == 0:
            metrics = evaluate(processor, model, head, dev, small=True)
            row = {"step": step, **stripped(metrics)}
            history.append(row)
            print(json.dumps(row), flush=True)
            quality = (min(metrics["stop_auc"] - .80,
                           metrics["instruction_swap_accuracy"] - .75,
                           metrics["balanced_progress_accuracy"] - .75,
                           metrics["regression_accuracy"] - .60),
                       metrics["stop_auc"] + metrics["instruction_swap_accuracy"] +
                       metrics["balanced_progress_accuracy"], -step)
            if best is None or quality > best[0]:
                best = (quality, step,
                        {k: v.detach().cpu().clone() for k, v in
                         get_peft_model_state_dict(model).items()},
                        copy.deepcopy({k: v.detach().cpu() for k, v in
                                       head.state_dict().items()}))
    if args.smoke:
        print(json.dumps({"smoke_updates": max_steps // grad_accum,
                          "lora_parameters": trainable,
                          "fit_expert": len(fit["expert"]),
                          "fit_policy": len(fit["policy"]),
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no checkpoint selected")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    full_dev = evaluate(processor, model, head, dev, small=False)
    threshold = choose_threshold(full_dev["positive_logits"],
                                 full_dev["negative_logits"])
    report = {"schema": "history_grounding_lora_development_v1",
              "seed": seed, "selected_step": best[1],
              "training_microsteps": max_steps, "grad_accum": grad_accum,
              "lora_parameters": trainable,
              "fit_expert": len(fit["expert"]), "fit_policy": len(fit["policy"]),
              "development_expert": len(dev["expert"]),
              "development_policy": len(dev["policy"]),
              "source_sha256": {
                  "scene_split": fit["scene_split_sha256"],
                  "expert_manifest": fit["expert_manifest_sha256"],
                  "policy_manifest": fit["policy_manifest_sha256"]},
              "development": stripped(full_dev),
              "development_stop_threshold": threshold,
              "checkpoint_selection": "small fixed development subset only",
              "history": history,
              "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "history_grounding_lora_seed11_v1",
                "adapter": best[2], "head": best[3],
                "selected_step": best[1],
                "source_sha256": report["source_sha256"],
                "model_config_sha256": digest(args.model / "config.json"),
                "development_stop_threshold": threshold["threshold"]},
               args.output / "adapter_head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1],
                      "development": report["development"],
                      "development_stop_threshold": threshold}, indent=2), flush=True)


if __name__ == "__main__":
    main()
