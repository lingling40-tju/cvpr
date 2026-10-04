"""Train a two-view, order-antisymmetric local progress reward model.

The model jointly sees before/after RGB views and the natural instruction.
Distance labels are used only for sampling and supervised loss. The audit
and val-unseen splits are absent. This is a new representation hypothesis
after independent-state potential subtraction failed development gating.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time

from PIL import Image
import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from verl.utils.dataset.vision_utils import process_image

from history_grounding_lora import digest, load_model, load_part
from train_policy_progress_lora import (choose, fixed_subset, gate,
                                        index_experts, indexed, pair_indices)


MAX_PIXELS = 76800
PAIR_PROMPT = (
    "Navigation instruction: {instruction}\n"
    "Before observation: <image>\n"
    "After observation: <image>\n"
    "Compare the two views for progress toward the instruction's destination."
)


def frame_path(item: dict, index: int) -> Path:
    record = item["record"]
    if not 0 <= index <= len(record["turns"]):
        raise ValueError("invalid turn-boundary index")
    relative = (record["initial_image"] if index == 0 else
                record["turns"][index - 1]["image"])
    return item["root"] / relative


def image(path: Path) -> Image.Image:
    with Image.open(path) as stream:
        raw = stream.convert("RGB")
    try:
        buffer = io.BytesIO()
        raw.save(buffer, format="PNG")
        processed = process_image({"bytes": buffer.getvalue(),
                                   "max_pixels": MAX_PIXELS,
                                   "min_pixels": 1024})
        return processed
    finally:
        raw.close()


def build_inputs(processor, item: dict, before: int, after: int,
                 instruction: str, reverse: bool) -> dict:
    if before >= after:
        raise ValueError("expected increasing physical turn indices")
    paths = (frame_path(item, after), frame_path(item, before)) if reverse else \
            (frame_path(item, before), frame_path(item, after))
    images = [image(path) for path in paths]
    try:
        message = {"role": "user", "content": PAIR_PROMPT.format(
            instruction=instruction.strip())}
        prompt = processor.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True)
        prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", prompt,
                        flags=re.S)
        if prompt.count("<image>") != 2:
            raise ValueError("pair prompt image count changed")
        prompt = prompt.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = processor(text=[prompt], images=images, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > 16000:
            raise ValueError("two-view context exceeds 16000 tokens")
        return inputs
    finally:
        for frame in images:
            if hasattr(frame, "close"):
                frame.close()


def score_pair(processor, model, head, item: dict, before: int, after: int,
               instruction: str, reverse: bool) -> torch.Tensor:
    inputs = build_inputs(processor, item, before, after,
                          instruction, reverse).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    _, value = head(output.logits[0, -1])
    return value


def pair_margin(processor, model, head, item: dict, before: int,
                after: int, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
    forward = score_pair(processor, model, head, item, before, after,
                         instruction, False)
    reverse = score_pair(processor, model, head, item, before, after,
                         instruction, True)
    return (forward - reverse) / 2, (forward + reverse) / 2


def local_loss(processor, model, head, item: dict, after: int,
               kind: str) -> torch.Tensor:
    instruction = item["record"]["instruction"].strip()
    margin, symmetry = pair_margin(processor, model, head, item,
                                   after - 1, after, instruction)
    meters = item["distances"][after - 1] - item["distances"][after]
    regularizer = .1 * symmetry.square()
    if kind == "stationary":
        if abs(meters) >= .1:
            raise ValueError("stationary label mismatch")
        return (F.smooth_l1_loss(margin, torch.zeros_like(margin), beta=.1) +
                F.relu(margin - .05) + regularizer)
    sign = 1.0 if kind == "forward" else -1.0
    if sign * meters < 1.0:
        raise ValueError("signed label mismatch")
    target = torch.as_tensor(math.tanh(meters / 3.0), dtype=margin.dtype,
                             device=margin.device)
    return (F.softplus(.2 - sign * margin) +
            .5 * F.smooth_l1_loss(margin, target, beta=.15) + regularizer)


def instruction_loss(processor, model, head, item: dict) -> torch.Tensor:
    if not item["safe_swap"]:
        raise ValueError("unsafe instruction swap")
    record = item["record"]
    end = len(record["turns"])
    correct, correct_symmetry = pair_margin(
        processor, model, head, item, 0, end, record["instruction"].strip())
    wrong, wrong_symmetry = pair_margin(
        processor, model, head, item, 0, end,
        record["wrong_instruction"].strip())
    return (F.softplus(.2 - correct) +
            F.softplus(.2 - (correct - wrong)) +
            .05 * (correct_symmetry.square() + wrong_symmetry.square()))


def evaluate(processor, model, head, data: dict, *, small: bool) -> dict:
    model.eval()
    head.eval()
    policies = data["policy"]
    experts = [item for item in data["expert"] if item["safe_swap"]]
    if small:
        policies = fixed_subset(policies, 96, "joint-pair-policy-dev")
        experts = fixed_subset(experts, 48, "joint-pair-expert-dev")
    hit = {kind: 0 for kind in ("forward", "backward")}
    total = {kind: 0 for kind in hit}
    by_scene = {kind: defaultdict(list) for kind in hit}
    episode_classes = {kind: set() for kind in
                       ("forward", "backward", "stationary")}
    stationary_values, forward_values = [], []
    swap_correct = 0
    with torch.inference_mode():
        for item in policies:
            record = item["record"]
            instruction = record["instruction"].strip()
            scene, eid = record["scene_id"], str(record["episode_id"])
            for kind, afters in pair_indices(item).items():
                for after in afters:
                    margin, _ = pair_margin(processor, model, head, item,
                                            after - 1, after, instruction)
                    value = float(margin)
                    episode_classes[kind].add(eid)
                    if kind == "stationary":
                        stationary_values.append(value)
                    elif kind == "forward":
                        correct = value > 0
                        hit[kind] += correct
                        total[kind] += 1
                        by_scene[kind][scene].append(correct)
                        forward_values.append(value)
                    else:
                        correct = value < 0
                        hit[kind] += correct
                        total[kind] += 1
                        by_scene[kind][scene].append(correct)
        for item in experts:
            record = item["record"]
            end = len(record["turns"])
            correct, _ = pair_margin(processor, model, head, item, 0, end,
                                     record["instruction"].strip())
            wrong, _ = pair_margin(processor, model, head, item, 0, end,
                                   record["wrong_instruction"].strip())
            swap_correct += float(correct) > float(wrong)
    if not all(total.values()) or not stationary_values or not experts or \
            not all(math.isfinite(x) for x in stationary_values + forward_values):
        raise ValueError("missing or invalid development pair class")
    ordered = sorted(stationary_values)
    threshold = max(0.0, ordered[min(len(ordered) - 1,
                                     math.ceil(.9 * len(ordered)) - 1)])
    accuracy = {kind: hit[kind] / total[kind] for kind in hit}
    macros = {kind: sum(sum(values) / len(values) for values in
                        by_scene[kind].values()) / len(by_scene[kind])
              for kind in hit}
    return {
        "policy_trajectories": len(policies),
        "expert_swap_trajectories": len(experts),
        "pair_counts": {**total, "stationary": len(stationary_values)},
        "unique_episodes_by_class": {kind: len(ids) for kind, ids in
                                      episode_classes.items()},
        "accuracy": accuracy, "scene_macro_accuracy": macros,
        "balanced_direction_accuracy": .5 * sum(accuracy.values()),
        "instruction_gain_preference": swap_correct / len(experts),
        "instruction_pairs": len(experts),
        "stationary_threshold_selected_on_development": threshold,
        "stationary_false_positive_rate": sum(x > threshold for x in
                                               stationary_values) / len(ordered),
        "forward_recall_at_stationary_threshold": sum(x > threshold for x in
                                                       forward_values) / len(forward_values),
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
    expert_index = index_experts(safe_experts)
    by_class = {kind: indexed(fit["policy"], kind) for kind in
                ("forward", "backward", "stationary")}
    processor, model, head, trainable = load_model(args.model)
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
        kind = ("forward", "backward", "stationary", "instruction")[(step - 1) % 4]
        if kind == "instruction":
            item = choose(expert_index, rng)
            loss = instruction_loss(processor, model, head, item)
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
        print(json.dumps({"smoke_updates": 1, "lora_parameters": trainable,
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
        "schema": "joint_pair_progress_lora_development_v1",
        "seed": seed, "microsteps": steps, "gradient_accumulation": accumulation,
        "lora_parameters": trainable, "selected_step": best[1],
        "source_sha256": {"scene_split": digest(args.scene_split),
                          "expert_manifest": digest(args.expert_manifest),
                          "policy_manifest": digest(args.policy_manifest),
                          "model_config": digest(args.model / "config.json")},
        "prompt": PAIR_PROMPT,
        "fit_policy_trajectories": len(fit["policy"]),
        "fit_safe_expert_trajectories": len(safe_experts),
        "selected_small_development": best[4],
        "small_development_gate": gate(best[4], preliminary=True),
        "full_development": full_dev,
        "full_development_gate": gate(full_dev, preliminary=False)
                                 if full_dev is not None else None,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "interpretation": "Train-scene development only; no audit or navigation result.",
    }
    torch.save({"schema": "joint_pair_progress_lora_v1",
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
