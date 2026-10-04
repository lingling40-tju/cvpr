"""Shared, label-isolated data and scoring for history-grounded LoRA.

The model input is the original ActiveVLN image/action/instruction prompt.
Geodesic distances select fit pairs and evaluate ranks, but are never
serialized into a prompt or otherwise passed to the visual-language model.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re

import torch
from torch import nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlnce_server.prompt import (SYSTEM_PROMPT_NO_THINK, action_template,
                                format_prompt, init_observation_template)


LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
        "target_modules": ["q_proj", "v_proj"]}
MAX_PIXELS = 76800
FORMAT = format_prompt["no_think_no_tag"](
    max_actions_per_step=3, action_sep=",", add_example=False)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rid(plan: dict) -> str:
    return f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"


def load_part(part: str, scene_split_path: Path, expert_manifest_path: Path,
              expert_labels_path: Path, expert_root: Path,
              policy_manifest_path: Path, policy_root: Path,
              policy_audit_path: Path) -> dict:
    split = json.loads(scene_split_path.read_text())
    expert_manifest = json.loads(expert_manifest_path.read_text())
    labels = json.loads(expert_labels_path.read_text())
    policy_manifest = json.loads(policy_manifest_path.read_text())
    policy_audit = json.loads(policy_audit_path.read_text())
    if split["schema"] != "stop_history_lora_scene_split_v1" or \
            split["label_audit_sha256"] != digest(expert_labels_path) or \
            labels["manifest_sha256"] != digest(expert_manifest_path) or \
            policy_manifest["scene_split_sha256"] != digest(scene_split_path) or \
            policy_audit["manifest_sha256"] != digest(policy_manifest_path) or \
            not policy_audit["regression_sample_gate"]["passed"]:
        raise ValueError("source hashes or regression gate invalid")
    scene_allow = set(split["scene_split"][part])
    old_expert_part = {str(row["episode_id"]): name
                       for name, rows in expert_manifest["selected"].items()
                       for row in rows}
    label_by_id = {str(row["episode_id"]): row
                   for rows in labels["labels"].values() for row in rows}
    expert = []
    for selection in split["selected"][part]:
        eid = str(selection["episode_id"])
        old_part = old_expert_part[eid]
        root = expert_root / old_part
        record = json.loads((root / "records" / f"{eid}.json").read_text())
        label = label_by_id[eid]
        if record["scene_id"] not in scene_allow or \
                record["manifest_sha256"] != digest(expert_manifest_path) or \
                record["turn_count"] > 12 or \
                bool(label["safe_wrong_instruction"]) != \
                bool(selection["safe_wrong_instruction"]):
            raise ValueError(f"invalid expert record {part}/{eid}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        if distances[-1] > 3.0 or distances[0] < 3.5:
            raise ValueError(f"invalid expert STOP labels {part}/{eid}")
        expert.append({"record": record, "root": root,
                       "distances": distances,
                       "safe_swap": bool(selection["safe_wrong_instruction"])})
    policy = []
    for plan in policy_manifest["selected"][part]:
        record = json.loads((policy_root / part / "records" /
                             f"{rid(plan)}.json").read_text())
        if record["scene_id"] not in scene_allow or \
                record["manifest_sha256"] != digest(policy_manifest_path) or \
                record["record_id"] != rid(plan):
            raise ValueError(f"invalid policy record {part}/{rid(plan)}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        policy.append({"record": record, "root": policy_root / part,
                       "distances": distances,
                       "pairs": signed_pairs(distances)})
    if len(expert) != len(split["selected"][part]) or \
            len(policy) != len(policy_manifest["selected"][part]):
        raise ValueError(f"incomplete {part} data")
    return {"expert": expert, "policy": policy,
            "scenes": sorted(scene_allow),
            "scene_split_sha256": digest(scene_split_path),
            "policy_manifest_sha256": digest(policy_manifest_path),
            "expert_manifest_sha256": digest(expert_manifest_path)}


def signed_pairs(distances: list[float]) -> dict[str, list[int]]:
    forward, backward = [], []
    for after in range(1, len(distances)):
        delta = distances[after] - distances[after - 1]
        if delta <= -1.0:
            forward.append(after)
        elif delta >= 1.0:
            backward.append(after)
    return {"forward": forward, "backward": backward}


def sampling_index(items: list[dict], category: str = "") -> dict[str, list]:
    by_scene = defaultdict(list)
    for item in items:
        if category and not item["pairs"][category]:
            continue
        by_scene[item["record"]["scene_id"]].append(item)
    if not by_scene:
        raise ValueError("empty sampling category")
    return dict(by_scene)


def _image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        raw = source.convert("RGB")
    processed = fetch_image({"image": raw, "max_pixels": MAX_PIXELS,
                             "min_pixels": 1024})
    if processed is not raw:
        raw.close()
    return processed


def _user(frame: Image.Image, instruction: str, initial: bool) -> dict:
    template = init_observation_template if initial else action_template
    full = template(observation="<image>", instruction=instruction) + "\n" + FORMAT
    if full.count("<image>") != 1:
        raise ValueError("unexpected image placeholder")
    prefix, suffix = full.split("<image>")
    return {"role": "user", "content": [
        {"type": "text", "text": prefix},
        {"type": "image", "image": frame},
        {"type": "text", "text": suffix}]}


def _policy_inputs(processor, item: dict, count: int,
                   instruction: str) -> dict:
    """Mirror parallel_env_vlnce's stripped prompt and image processing."""
    from verl.utils.dataset.vision_utils import process_image

    record = item["record"]
    paths = [record["initial_image"]] + [turn["image"] for turn in
                                                 record["turns"][:count]]
    frames = []
    for path in paths:
        with Image.open(item["root"] / path) as source:
            raw = source.convert("RGB")
        try:
            buffer = io.BytesIO()
            raw.save(buffer, format="PNG")
            frames.append(process_image({"bytes": buffer.getvalue(),
                                         "max_pixels": MAX_PIXELS,
                                         "min_pixels": 1024}))
        finally:
            raw.close()
    try:
        messages = [{"role": "user", "content":
                     init_observation_template("<image>", instruction) +
                     "\n" + FORMAT}]
        for index in range(count):
            response = record["turns"][index]["assistant_response"]
            if "stop" in response.lower():
                raise ValueError("STOP in observation history")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                             action_template("<image>", instruction) +
                             "\n" + FORMAT})
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", prompt,
                        flags=re.S)
        prompt = prompt.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = processor(text=[prompt], images=frames, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > 16000:
            raise ValueError("history context exceeds 16000 tokens")
        return inputs
    finally:
        for frame in frames:
            if hasattr(frame, "close"):
                frame.close()


def build_inputs(processor, item: dict, count: int,
                 instruction: str | None = None,
                 include_system: bool = True) -> dict:
    record = item["record"]
    if count < 0 or count > len(record["turns"]):
        raise ValueError("invalid history state index")
    instruction = instruction or record["instruction"]
    if not include_system:
        return _policy_inputs(processor, item, count, instruction)
    paths = [record["initial_image"]] + [turn["image"] for turn in
                                                 record["turns"][:count]]
    frames = [_image(item["root"] / path) for path in paths]
    try:
        messages = ([{"role": "system", "content": [{
            "type": "text", "text": SYSTEM_PROMPT_NO_THINK["r2r"]}]}]
                    if include_system else [])
        messages.append(_user(frames[0], instruction, True))
        for index in range(count):
            response = record["turns"][index]["assistant_response"]
            if "stop" in response.lower():
                raise ValueError("STOP in observation history")
            messages.append({"role": "assistant", "content": [{
                "type": "text", "text": response}]})
            messages.append(_user(frames[index + 1], instruction, False))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > 16000:
            raise ValueError("history context exceeds 16000 tokens")
        return inputs
    finally:
        for frame in frames:
            frame.close()


class RewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, 128), nn.GELU(),
                                 nn.Linear(128, 2))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(hidden.float())
        return scores[..., 0], torch.tanh(scores[..., 1])


def load_model(path: Path):
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True,
                                              use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, local_files_only=True)
    base.config.use_cache = False
    # This reward model reads the assistant-prefix state; the 151k-token
    # vocabulary projection is unused and otherwise dominates each pass.
    base.lm_head = nn.Identity()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(**LORA)).cuda()
    model.enable_input_require_grads()
    head = RewardHead(base.config.hidden_size).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable < 100_000 or trainable > 30_000_000:
        raise ValueError(f"unexpected LoRA size: {trainable}")
    return processor, model, head, trainable


def score(processor, model, head, item: dict, count: int,
          instruction: str | None = None,
          include_system: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = build_inputs(processor, item, count, instruction,
                          include_system=include_system).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    hidden = output.logits[0, -1]  # Identity LM head exposes the last state.
    return head(hidden)
