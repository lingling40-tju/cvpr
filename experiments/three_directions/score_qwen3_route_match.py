"""Frozen six-view, two-instruction route comparison with local Qwen3-VL.

Writes no source RGB or instruction text. Two prompt orderings cancel
A/B letter preference. This is an offline diagnostic, not a reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from prepare_clause_alignment_manifest import digest


SYSTEM = ("You compare indoor navigation instructions with an observed route. "
          "Use only the ordered photos and the two candidate instructions. "
          "Choose the instruction that matches the route and its later destination. "
          "Your next response must begin with exactly A or B. Do not explain.")
INTRO = ("The six photos below are ordered snapshots sampled along one indoor "
         "navigation route. Two instructions start from the same place but lead "
         "to different destinations. Which instruction matches this route?\n")
ENDING = "\nRespond with exactly A or B."


def prompt_hash(messages: list[dict]) -> str:
    # Hash text and frame indices without serializing RGB.
    serializable = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            serializable.append([item["text"] if item["type"] == "text" else "[image]"
                                 for item in content])
        else:
            serializable.append(content)
    return hashlib.sha256(json.dumps(serializable, ensure_ascii=False).encode()).hexdigest()


def load_frames(record: dict, part_root: Path) -> list[Image.Image]:
    if len(record["frames"]) != 6:
        raise ValueError("expected six ordered RGB frames")
    frames = []
    for frame in record["frames"]:
        path = (part_root / frame["image"]).resolve()
        if not path.is_relative_to(part_root):
            raise ValueError("image path escapes source root")
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((448, 448), Image.Resampling.BICUBIC)
        frames.append(image)
    return frames


def message(frames: list[Image.Image], instruction_a: str,
            instruction_b: str) -> list[dict]:
    content = [{"type": "text", "text": INTRO}]
    for i, frame in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Photo {i} of 6:"})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text":
                    f"\nInstruction A: {instruction_a}\nInstruction B: {instruction_b}{ENDING}"})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def logit_margin(model, processor, frames: list[Image.Image],
                 instruction_a: str, instruction_b: str,
                 a_token: int, b_token: int) -> tuple[float, str, int]:
    messages = message(frames, instruction_a, instruction_b)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    if int(inputs["input_ids"].shape[1]) > 16000:
        raise ValueError("route matching context too long")
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    logits = output.logits[0, -1].float()
    result = float(logits[a_token].item() - logits[b_token].item())
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError("nonfinite A/B margin")
    return result, prompt_hash(messages), int(inputs["input_ids"].shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "calibration"), default="calibration")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.shards != 4:
        raise ValueError("exactly four prespecified shards required")
    manifest = json.loads(args.manifest.read_text())
    ordinal = json.loads(args.ordinal_manifest.read_text())
    if manifest["schema"] != "clause_alignment_manifest_v1" or \
            manifest["ordinal_manifest_sha256"] != digest(args.ordinal_manifest):
        raise ValueError("source manifest changed")
    pairs = ordinal["subsets"][args.part]["pairs"]
    if len(pairs) != {"fit": 128, "calibration": 32}[args.part]:
        raise ValueError("natural pair count changed")
    rows = {row["episode_id"]: row for row in manifest["selected"][args.part]}
    selected = pairs[args.shard::args.shards]
    if len(selected) != {"fit": 32, "calibration": 8}[args.part]:
        raise ValueError("unbalanced pair shard")
    expected_model_files = ["config.json", "tokenizer.json", "preprocessor_config.json",
                            "model.safetensors.index.json"] + [
        f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    sha_lines = args.model_hashes.read_text().splitlines()
    if [line.split()[-1] for line in sha_lines] != expected_model_files:
        raise ValueError("model weight fingerprint file changed")
    source_root = (args.record_root / args.part).resolve()
    records = {}
    for pair in selected:
        for eid in (pair["left"], pair["right"]):
            row = rows[eid]
            path = source_root / "records" / f"{eid}.json"
            if digest(path) != row["record_sha256"]:
                raise ValueError(f"source hash mismatch: {eid}")
            record = json.loads(path.read_text())
            if record["episode_id"] != eid or record["scene_id"] != pair["scene"] or \
                    len(record["frames"]) != 6:
                raise ValueError(f"source identity mismatch: {eid}")
            records[eid] = record
    started = time.time()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    a_ids = processor.tokenizer.encode("A", add_special_tokens=False)
    b_ids = processor.tokenizer.encode("B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError("A/B not single tokens")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="auto",
        local_files_only=True).eval()
    results = []
    for pair_index, pair in enumerate(selected):
        left, right = pair["left"], pair["right"]
        entry = {"episode_ids": [left, right], "scene_id": pair["scene"],
                 "routes": []}
        for route_index, eid in enumerate((left, right)):
            frames = load_frames(records[eid], source_root)
            try:
                left_text, right_text = records[left]["instruction"], records[right]["instruction"]
                first, hash_first, tokens_first = logit_margin(
                    model, processor, frames, left_text, right_text, a_ids[0], b_ids[0])
                swapped, hash_swapped, tokens_swapped = logit_margin(
                    model, processor, frames, right_text, left_text, a_ids[0], b_ids[0])
                correct_first = first if route_index == 0 else -first
                correct_swapped = -swapped if route_index == 0 else swapped
                entry["routes"].append({"episode_id": eid,
                                         "correct_margin_first": correct_first,
                                         "correct_margin_swapped": correct_swapped,
                                         "correct_margin_average": (correct_first + correct_swapped) / 2,
                                         "prompt_sha256": [hash_first, hash_swapped],
                                         "input_tokens": [tokens_first, tokens_swapped]})
            finally:
                for frame in frames:
                    frame.close()
            if args.smoke_only:
                break
        results.append(entry)
        if args.smoke_only:
            break
        print(f"shard {args.shard} pair {pair_index + 1}/{len(selected)} "
              f"elapsed_s={time.time() - started:.1f}", flush=True)
        # Save every complete pair so a preempted shard can be audited; the
        # final summary still requires exactly eight pairs and 32 queries.
        partial = args.output.with_suffix(".partial.json")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(results, indent=2) + "\n")
    payload = {"schema": "qwen3_route_match_shard_v1", "part": args.part,
               "smoke_only": args.smoke_only,
               "shard": args.shard, "shards": args.shards, "pairs": len(results),
               "routes": sum(len(item["routes"]) for item in results),
               "queries": 2 * sum(len(item["routes"]) for item in results),
               "source_sha256": {"manifest": digest(args.manifest),
                                  "ordinal_manifest": digest(args.ordinal_manifest),
                                  "model_hash_file": digest(args.model_hashes),
                                  "model_config": digest(args.model / "config.json"),
                                  "tokenizer": digest(args.model / "tokenizer.json")},
               "token_ids": {"A": a_ids[0], "B": b_ids[0]},
               "max_image_edge": 448, "max_input_tokens": 16000,
               "decoding": "first assistant token, A minus B raw logits, two orderings; no sampling",
               "prompt_sha256": hashlib.sha256((SYSTEM + INTRO + ENDING).encode()).hexdigest(),
               "elapsed_seconds": time.time() - started, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, args.output)
    print(json.dumps({"pairs": payload["pairs"], "routes": payload["routes"],
                      "queries": payload["queries"],
                      "elapsed_seconds": payload["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
