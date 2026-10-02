"""Score frozen navigation-SFT STOP readiness using real multi-turn history.

Terminal scores are computed first; initial scores are a separate optional
phase, so an unpromising pair-ranking screen does not consume extra GPU time.
The query deliberately excludes the model's terminal STOP answer and labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT_VERSION = "vlnce_multiturn_no_system_stop_omitted_v2"
USER_SUFFIX = "\nDecide your next action. \nYou can take up to 3 actions at a time, separated by ','. "


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def user_content(instruction: str, initial: bool) -> str:
    prefix = "[Initial Observation]:\n" if initial else "After that, the observation is:\n"
    return prefix + "<image>\nInstruction: " + instruction + USER_SUFFIX


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        raw = source.convert("RGB")
    image = fetch_image({"image": raw, "max_pixels": 76800,
                         "min_pixels": 1024})
    if image is not raw:
        raw.close()
    return image


def score(model, processor, record: dict, root: Path, token_ids: list[int],
          phase: str) -> dict:
    images = []
    try:
        initial = load_image(root / record["initial_image"])
        images.append(initial)
        messages = [{"role": "user",
                     "content": user_content(record["instruction"], True)}]
        if phase == "terminal":
            for turn in record["turns"]:
                if "stop" in turn["assistant_response"].lower():
                    raise ValueError("terminal STOP leaked into context")
                frame = load_image(root / turn["image"])
                images.append(frame)
                messages.extend([
                    {"role": "assistant", "content": turn["assistant_response"]},
                    {"role": "user", "content": user_content(
                        record["instruction"], False)},
                ])
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace("<image>",
                                "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = processor(text=[prompt], images=images,
                           return_tensors="pt").to(model.device)
        context_tokens = int(inputs["input_ids"].shape[1])
        if context_tokens > 16000:
            raise ValueError(f"context exceeds 16000 tokens: {context_tokens}")
        with torch.inference_mode():
            result = model(**inputs, use_cache=False)
            logits = result.logits[0, -1].float()
            margin = (logits[token_ids[0]] -
                      torch.logsumexp(logits[token_ids[1:]], dim=0)).item()
        if not torch.isfinite(torch.tensor(margin)):
            raise ValueError("nonfinite STOP margin")
        return {"stop_margin": margin, "context_tokens": context_tokens,
                "images": len(images), "assistant_turns": len(messages) // 2}
    finally:
        for image in images:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("terminal", "initial"), default="terminal")
    parser.add_argument("--split", choices=("development", "audit"))
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    probe = json.loads(args.probe_manifest.read_text())
    collection = json.loads((args.records_root / "summary.json").read_text())
    pairs = [item for item in probe["pairs"]
             if args.split is None or item["split"] == args.split]
    pairs = pairs[:args.limit_pairs or None]
    if not pairs or args.limit_pairs < 0:
        raise ValueError("empty or invalid pair selection")
    if collection["probe_manifest_sha256"] != digest(args.probe_manifest) or \
            collection["errors"]:
        raise ValueError("full-history collection incomplete")
    args.output_root.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    token_ids = []
    for word in ("stop", "move", "turn"):
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"action prefix is not one token: {word}")
        token_ids.append(ids[0])
    model_hash = digest(args.model / "config.json")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    begun = time.time()
    for number, pair in enumerate(pairs, 1):
        for role in ("success", "failure"):
            record_id = pair["pair_id"] + "_" + role
            record_path = args.records_root / "records" / f"{record_id}.json"
            record = json.loads(record_path.read_text())
            if record["record_id"] != record_id or \
                    record["split"] != pair["split"] or \
                    record["scene_id"] != pair["scene_id"]:
                raise ValueError(f"record/probe mismatch {record_id}")
            out = args.output_root / f"{record_id}_{args.phase}.json"
            if out.is_file():
                cached = json.loads(out.read_text())
                if cached["record_sha256"] == digest(record_path) and \
                        cached["model_config_sha256"] == model_hash and \
                        cached["prompt_version"] == PROMPT_VERSION:
                    continue
            result = score(model, processor, record, args.records_root,
                           token_ids, args.phase)
            payload = {"record_id": record_id, "phase": args.phase,
                       "split": pair["split"], "scene_id": pair["scene_id"],
                       "record_sha256": digest(record_path),
                       "model_config_sha256": model_hash,
                       "prompt_version": PROMPT_VERSION, **result}
            temp = out.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temp, out)
        if number % 10 == 0 or number == len(pairs):
            print(f"{args.phase} pairs {number}/{len(pairs)} elapsed_s={time.time()-begun:.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
