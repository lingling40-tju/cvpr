"""Action-conditioned, start-anchored observation-only progress probe.

The model sees the instruction, the start/before/after RGB observations, and
the executed action text. Simulator distances define fit/development labels
only. Neither the old audit nor val-unseen is loaded by this program.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part
from train_joint_pair_progress_lora import frame_path, image
from train_policy_progress_lora import (choose, fixed_subset, gate,
                                        index_experts, indexed, pair_indices)


PROMPT = (
    "Navigation instruction: {instruction}\n"
    "Starting view: <image>\n"
    "View before the action: <image>\n"
    "Executed action sequence: {actions}\n"
    "View after the action: <image>\n"
    "Score whether this action made progress toward the instruction's goal."
)


def action_text(item: dict, before: int, after: int) -> str:
    turns = item["record"]["turns"]
    if not 0 <= before < after <= len(turns):
        raise ValueError("invalid action span")
    actions = [str(turn["assistant_response"]).strip()
               for turn in turns[before:after]]
    if not all(actions):
        raise ValueError("missing action response")
    return " ; ".join(actions)


def build_inputs(processor, item: dict, before: int, after: int,
                 instruction: str | None = None):
    record = item["record"]
    paths = [frame_path(item, 0), frame_path(item, before),
             frame_path(item, after)]
    frames = [image(path) for path in paths]
    try:
        message = {"role": "user", "content": PROMPT.format(
            instruction=(instruction or record["instruction"]).strip(),
            actions=action_text(item, before, after))}
        text = processor.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True)
        text = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", text,
                      flags=re.S)
        if text.count("<image>") != 3:
            raise ValueError("action-memory prompt must contain three images")
        text = text.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = processor(text=[text], images=frames, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > 16000:
            raise ValueError("action-memory context exceeds 16000 tokens")
        return inputs
    finally:
        for frame in frames:
            frame.close()


def score(processor, model, head, item: dict, before: int, after: int,
          instruction: str | None = None) -> torch.Tensor:
    inputs = build_inputs(processor, item, before, after, instruction).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    _, value = head(output.logits[0, -1])
    return value


def local_loss(processor, model, head, item: dict, after: int,
               kind: str) -> torch.Tensor:
    meters = item["distances"][after - 1] - item["distances"][after]
    value = score(processor, model, head, item, after - 1, after)
    if kind == "stationary":
        if abs(meters) >= .1:
            raise ValueError("stationary label mismatch")
        return F.smooth_l1_loss(value, torch.zeros_like(value), beta=.1) + \
            F.relu(value - .05)
    sign = 1.0 if kind == "forward" else -1.0
    if sign * meters < 1.0:
        raise ValueError("signed geodesic label mismatch")
    target = torch.as_tensor(math.tanh(meters / 3.0), dtype=value.dtype,
                             device=value.device)
    return F.softplus(.2 - sign * value) + \
        .5 * F.smooth_l1_loss(value, target, beta=.15)


def rank_loss(processor, model, head, item: dict,
              rng: random.Random) -> torch.Tensor:
    pairs = pair_indices(item)
    if not pairs["forward"] or not pairs["backward"]:
        raise ValueError("within-trajectory direction contrast unavailable")
    advance = rng.choice(pairs["forward"])
    retreat = rng.choice(pairs["backward"])
    positive = score(processor, model, head, item, advance - 1, advance)
    negative = score(processor, model, head, item, retreat - 1, retreat)
    return F.softplus(.3 - (positive - negative))


def instruction_loss(processor, model, head, item: dict) -> torch.Tensor:
    if not item["safe_swap"]:
        raise ValueError("unsafe same-start instruction swap")
    record = item["record"]
    end = len(record["turns"])
    correct = score(processor, model, head, item, 0, end,
                    record["instruction"])
    wrong = score(processor, model, head, item, 0, end,
                  record["wrong_instruction"])
    return F.softplus(.2 - correct) + \
        F.softplus(.2 - (correct - wrong))


def index_both(items: list[dict]) -> dict:
    by_scene = defaultdict(lambda: defaultdict(list))
    for item in items:
        pairs = pair_indices(item)
        if pairs["forward"] and pairs["backward"]:
            record = item["record"]
            by_scene[record["scene_id"]][str(record["episode_id"])].append(item)
    if not by_scene:
        raise ValueError("no within-trajectory forward/backward pairs")
    return by_scene


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
    episodes = {kind: set() for kind in
                ("forward", "backward", "stationary")}
    stationary, forward = [], []
    instruction_hits = 0
    with torch.inference_mode():
        for item in policies:
            record = item["record"]
            for kind, afters in pair_indices(item).items():
                for after in afters:
                    value = float(score(processor, model, head, item,
                                        after - 1, after))
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
            correct = float(score(processor, model, head, item, 0, end,
                                  record["instruction"]))
            wrong = float(score(processor, model, head, item, 0, end,
                                record["wrong_instruction"]))
            instruction_hits += correct > wrong
    if not all(total.values()) or not stationary or not forward or not experts:
        raise ValueError("missing development class")
    ordered = sorted(stationary)
    threshold = max(0.0, ordered[min(len(ordered) - 1,
                                     math.ceil(.9 * len(ordered)) - 1)])
    accuracy = {kind: hit[kind] / total[kind] for kind in hit}
    macro = {kind: sum(sum(values) / len(values) for values in
                       by_scene[kind].values()) / len(by_scene[kind])
             for kind in hit}
    return {
        "policy_trajectories": len(policies),
        "expert_swap_trajectories": len(experts),
        "pair_counts": {**total, "stationary": len(stationary)},
        "unique_episodes_by_class": {kind: len(ids)
                                     for kind, ids in episodes.items()},
        "accuracy": accuracy, "scene_macro_accuracy": macro,
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
                 "policy-audit", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prompt-check", action="store_true")
    args = parser.parse_args()
    seed, steps, accumulation = 11, (5 if args.smoke else 1000), 5
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit = load_part("fit", *common)
    dev = None if args.smoke or args.prompt_check else load_part("development", *common)
    if args.prompt_check:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(args.model),
                                                  local_files_only=True,
                                                  use_fast=False)
        item = next(row for row in fit["policy"] if row["record"]["turns"])
        inputs = build_inputs(processor, item, 0, 1)
        if inputs.get("pixel_values") is None or \
                not torch.isfinite(inputs["pixel_values"]).all():
            raise ValueError("invalid action-memory image tensors")
        print(json.dumps({"prompt_check": "passed",
                          "text_tokens": int(inputs["input_ids"].shape[1]),
                          "image_rows": int(inputs["pixel_values"].shape[0])}))
        return
    safe_experts = [item for item in fit["expert"] if item["safe_swap"]]
    expert_index = index_experts(safe_experts)
    by_class = {kind: indexed(fit["policy"], kind) for kind in
                ("forward", "backward", "stationary")}
    both = index_both(fit["policy"])
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
        kind = ("forward", "backward", "stationary", "instruction",
                "within_trajectory")[(step - 1) % 5]
        if kind == "instruction":
            loss = instruction_loss(processor, model, head,
                                    choose(expert_index, rng))
        elif kind == "within_trajectory":
            loss = rank_loss(processor, model, head, choose(both, rng), rng)
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
        if not args.smoke and step in (250, 500, 750, 1000):
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
                          "fit_both_direction_trajectories": sum(
                              len(rows) for episodes in both.values()
                              for rows in episodes.values()),
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no selected checkpoint")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    full_dev = (evaluate(processor, model, head, dev, small=False)
                if all(gate(best[4], preliminary=True).values()) else None)
    report = {
        "schema": "action_memory_progress_lora_development_v1",
        "interpretation": "Train-scene development only; no audit or navigation result.",
        "seed": seed, "microsteps": steps,
        "gradient_accumulation": accumulation,
        "lora_parameters": trainable, "selected_step": best[1],
        "source_sha256": {"scene_split": digest(args.scene_split),
                          "expert_manifest": digest(args.expert_manifest),
                          "policy_manifest": digest(args.policy_manifest),
                          "model_config": digest(args.model / "config.json")},
        "prompt": PROMPT,
        "fit_policy_trajectories": len(fit["policy"]),
        "fit_safe_expert_trajectories": len(safe_experts),
        "selected_small_development": best[4],
        "small_development_gate": gate(best[4], preliminary=True),
        "full_development": full_dev,
        "full_development_gate": gate(full_dev, preliminary=False)
                                 if full_dev is not None else None,
        "history": history, "elapsed_seconds": time.time() - started,
    }
    torch.save({"schema": "action_memory_progress_lora_v1",
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
