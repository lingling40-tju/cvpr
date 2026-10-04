"""Check the new STOP reward's initial prompt against the policy trainer.

Reads one fit-scene cached policy frame and the local processor only. No
model weights, simulator distance, development split, or audit split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor

from history_grounding_lora import FORMAT, MAX_PIXELS, build_inputs, digest, rid
from verl.workers.agent.parallel_env_vlnce import (
    _preprocess_multi_modal_inputs, _strip_system_block)
from vlnce_server.prompt import action_template, init_observation_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())["selected"]["fit"][0]
    root = args.policy_root / "fit"
    record = json.loads((root / "records" / (rid(plan) + ".json")).read_text())
    if record["record_id"] != rid(plan):
        raise ValueError("selected fit record identity mismatch")
    processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True, use_fast=False)
    item = {"record": record, "root": root}
    checks = []
    for count in sorted({0, 1, min(3, len(record["turns"]))}):
        proposed = build_inputs(processor, item, count,
                                include_system=False)
        observation = init_observation_template(
            observation="<image>", instruction=record["instruction"]) + \
            "\n" + FORMAT
        messages = [{"role": "user", "content": observation}]
        for turn in record["turns"][:count]:
            messages.append({"role": "assistant",
                             "content": turn["assistant_response"]})
            messages.append({"role": "user", "content":
                             action_template("<image>",
                                             record["instruction"]) +
                             "\n" + FORMAT})
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt = _strip_system_block(prompt)
        frames = []
        paths = [record["initial_image"]] + [turn["image"] for turn in
                                                record["turns"][:count]]
        for path in paths:
            with Image.open(root / path) as source:
                frames.append(source.convert("RGB"))
        _, token_ids, mm_inputs = _preprocess_multi_modal_inputs(
            prompt, processor, multi_modal_data={"image": frames},
            max_pixels=MAX_PIXELS, min_pixels=1024)
        trainer = {"input_ids": token_ids.unsqueeze(0),
                   "attention_mask": torch.ones_like(token_ids).unsqueeze(0),
                   **mm_inputs}
        keys = ("input_ids", "attention_mask", "image_grid_thw",
                "pixel_values")
        equal = {key: (bool((proposed[key] == trainer[key]).all())
                       if proposed[key].shape == trainer[key].shape else False)
                 if key in proposed and key in trainer else None
                 for key in keys}
        checks.append({"history_turns": count, "equal": equal,
                       "proposed_tokens": int(proposed["input_ids"].shape[1]),
                       "trainer_tokens": int(trainer["input_ids"].shape[1])})
        for frame in frames:
            if hasattr(frame, "close"):
                frame.close()
    result = {"schema": "policy_stop_prompt_parity_v1",
              "fit_record_id": record["record_id"], "checks": checks,
              "source_sha256": {
                  "manifest": digest(args.manifest),
                  "reward_prompt_helper": digest(
                      Path(__file__).with_name("history_grounding_lora.py")),
                  "policy_trainer_prompt": digest(Path(
                      _preprocess_multi_modal_inputs.__code__.co_filename))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(any(value is not True for value in check["equal"].values())
           for check in checks):
        raise ValueError("reward/policy prompt mismatch")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
