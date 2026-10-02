"""Cache train-only navigation-SFT multimodal state and STOP readiness.

The frozen Qwen2.5-VL-3B SFT model sees a current RGB view and the R2R
instruction in the navigation prompt. It produces one fused hidden vector
and a STOP-versus-move/turn first-token logit margin per sparse frame. No
generation, verifier labels, external API, or RL update is used.
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, "
    "turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, turn right 15 degrees, turn right 30 degrees, "
    "turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)
USER_SUFFIX = "Decide your next action. You can take up to 3 actions at a time, separated by ','. "


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_one(model, processor, image_path: Path, instruction: str,
               action_tokens: tuple[int, int, int]) -> tuple[torch.Tensor, float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Instruction: " + instruction + USER_SUFFIX},
        ]},
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        vector = output.hidden_states[-1][0, -1].float().cpu()
        logits = output.logits[0, -1].float()
        stop, move, turn = action_tokens
        margin = (logits[stop] - torch.logsumexp(logits[[move, turn]], dim=0)).item()
    image.close()
    if not torch.isfinite(vector).all() or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("nonfinite navigation-SFT output")
    return vector.to(torch.float16), float(margin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    collection = json.loads((args.records_root / "summary.json").read_text())
    if collection["pairs"] != len(manifest["pairs"]) or \
            collection["completed_trajectories"] != 2 * len(manifest["pairs"]) or \
            collection["errors"] or collection["manifest_sha256"] != digest(args.manifest):
        raise ValueError("source frame collection incomplete")
    ids = [pair["pair_id"] + "_" + role for pair in manifest["pairs"]
           for role in ("success", "failure")]
    if args.limit_records:
        if args.limit_records <= 0 or args.limit_records > len(ids):
            raise ValueError("invalid record limit")
        ids = ids[:args.limit_records]
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_root / "records"
    cache_dir.mkdir(exist_ok=True)
    config_hash = digest(args.model / "config.json")
    manifest_hash = digest(args.manifest)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    token_ids = []
    for name in ("stop", "move", "turn"):
        encoded = processor.tokenizer.encode(name, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action prefix {name} is not a single token")
        token_ids.append(encoded[0])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    start_time = time.time()
    all_vectors, all_margins = [], []
    for number, record_id in enumerate(ids, 1):
        record = json.loads((args.records_root / "records" /
                             f"{record_id}.json").read_text())
        if record["record_id"] != record_id or len(record["frames"]) != 4:
            raise ValueError(f"bad source record {record_id}")
        cache_path = cache_dir / f"{record_id}.pt"
        cached = None
        if cache_path.is_file():
            candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
            if candidate.get("record_id") == record_id and \
                    candidate.get("manifest_sha256") == manifest_hash and \
                    candidate.get("model_config_sha256") == config_hash and \
                    candidate["hidden"].shape == (4, model.config.hidden_size) and \
                    candidate["stop_margin"].shape == (4,) and \
                    torch.isfinite(candidate["hidden"]).all() and \
                    torch.isfinite(candidate["stop_margin"]).all():
                cached = candidate
        if cached is None:
            vectors, margins = [], []
            for frame in record["frames"]:
                vector, margin = encode_one(model, processor,
                                            args.records_root / frame["image"],
                                            record["instruction"], tuple(token_ids))
                vectors.append(vector)
                margins.append(margin)
            cached = {"record_id": record_id, "manifest_sha256": manifest_hash,
                      "model_config_sha256": config_hash,
                      "hidden": torch.stack(vectors),
                      "stop_margin": torch.tensor(margins, dtype=torch.float32)}
            temp = cache_path.with_suffix(".tmp")
            torch.save(cached, temp)
            os.replace(temp, cache_path)
        all_vectors.append(cached["hidden"])
        all_margins.append(cached["stop_margin"])
        if number % 20 == 0 or number == len(ids):
            print(f"states {number}/{len(ids)} elapsed_s={time.time()-start_time:.1f}",
                  flush=True)
    payload = {"schema": "navigation_sft_state_v1", "record_ids": ids,
               "manifest_sha256": manifest_hash,
               "model_config_sha256": config_hash,
               "model_name": args.model.name,
               "action_first_token_ids": dict(zip(("stop", "move", "turn"), token_ids)),
               "hidden": torch.stack(all_vectors),
               "stop_margin": torch.stack(all_margins)}
    output = args.output_root / "features.pt"
    torch.save(payload, output)
    summary = {"records": len(ids), "frames": 4 * len(ids),
               "manifest_sha256": manifest_hash,
               "model_config_sha256": config_hash,
               "elapsed_seconds": time.time() - start_time,
               "output": str(output)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
