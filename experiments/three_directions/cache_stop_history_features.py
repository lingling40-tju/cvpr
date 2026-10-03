"""Cache frozen navigation-SFT states for train-only history/STOP learning.

Uses the ActiveVLN server's actual system and per-observation templates.
Only histories of at most 12 turns enter this first screen. The goal
distance and success labels are never passed to the model. Cached .pt
files contain only tensors and safe primitive metadata; resume reads use
torch.load(weights_only=True).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import vlnce_server.prompt as prompt_module
from vlnce_server.prompt import (SYSTEM_PROMPT_NO_THINK, action_template,
                                format_prompt, init_observation_template)


PROMPT_VERSION = "vlnce_server_multiturn_expert_12turn_v1"
MAX_PIXELS = 76800


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        raw = source.convert("RGB")
    processed = fetch_image({"image": raw, "max_pixels": MAX_PIXELS,
                             "min_pixels": 1024})
    if processed is not raw:
        raw.close()
    return processed


def user_message(frame: Image.Image, instruction: str, initial: bool,
                 format_text: str) -> dict:
    template = init_observation_template if initial else action_template
    full = template(observation="<image>", instruction=instruction) + "\n" + format_text
    if full.count("<image>") != 1:
        raise ValueError("unexpected image placeholder")
    prefix, suffix = full.split("<image>")
    return {"role": "user", "content": [
        {"type": "text", "text": prefix},
        {"type": "image", "image": frame},
        {"type": "text", "text": suffix},
    ]}


def score(model, processor, frames: list[Image.Image], turns: list[dict],
          instruction: str, count: int, action_tokens: tuple[int, int, int],
          format_text: str) -> tuple[torch.Tensor, float, int]:
    messages = [{"role": "system", "content": [
        {"type": "text", "text": SYSTEM_PROMPT_NO_THINK["r2r"]}]}]
    messages.append(user_message(frames[0], instruction, True, format_text))
    for index in range(count):
        response = turns[index]["assistant_response"]
        if "stop" in response.lower():
            raise ValueError("STOP leaked into history")
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": response}]})
        messages.append(user_message(frames[index + 1], instruction,
                                     False, format_text))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    context_tokens = int(inputs["input_ids"].shape[1])
    if context_tokens > 16000:
        raise ValueError(f"context exceeds 16000 tokens: {context_tokens}")
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        vector = output.hidden_states[-1][0, -1].float().cpu()
        logits = output.logits[0, -1].float()
        stop, move, turn = action_tokens
        margin = (logits[stop] - torch.logsumexp(logits[[move, turn]], dim=0)).item()
    if not torch.isfinite(vector).all() or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("nonfinite SFT features")
    return vector.to(torch.float16), float(margin), context_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.label_audit.read_text())
    manifest_sha = digest(args.manifest)
    if (manifest["schema"] != "stop_history_expert_manifest_v1" or
            audit["schema"] != "stop_history_train_only_label_audit_v1" or
            audit["manifest_sha256"] != manifest_sha):
        raise ValueError("manifest/label audit mismatch")
    label_by_id = {str(row["episode_id"]): row for row in audit["labels"][args.part]}
    selected = [plan for plan in manifest["selected"][args.part]
                if label_by_id[str(plan["episode_id"])]["within_12_turns"]]
    if args.limit < 0:
        raise ValueError("negative limit")
    selected = selected[:args.limit or None]
    output = args.output_root / args.part
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "records"
    cache.mkdir(exist_ok=True)
    model_hash = digest(args.model / "config.json")
    prompt_hash = digest(Path(prompt_module.__file__))
    processor = AutoProcessor.from_pretrained(str(args.model),
                                              local_files_only=True,
                                              use_fast=False)
    action_tokens = []
    for word in ("stop", "move", "turn"):
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"non-single action prefix {word}")
        action_tokens.append(ids[0])
    format_text = format_prompt["no_think_no_tag"](
        max_actions_per_step=3, action_sep=",", add_example=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    begun = time.time()
    completed = []
    errors = []
    for number, plan in enumerate(selected, 1):
        episode_id = str(plan["episode_id"])
        record_path = args.records_root / args.part / "records" / f"{episode_id}.json"
        record = json.loads(record_path.read_text())
        label = label_by_id[episode_id]
        if (record["manifest_sha256"] != manifest_sha or
                record["episode_id"] != episode_id or
                record["turn_count"] > 12 or
                not label["stop_labels_valid"]):
            raise ValueError(f"invalid source record/label {episode_id}")
        record_sha = digest(record_path)
        cache_path = cache / f"{episode_id}.pt"
        if cache_path.is_file():
            previous = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (previous.get("record_sha256") == record_sha and
                    previous.get("manifest_sha256") == manifest_sha and
                    previous.get("model_config_sha256") == model_hash and
                    previous.get("prompt_sha256") == prompt_hash and
                    previous.get("prompt_version") == PROMPT_VERSION and
                    previous["hidden"].shape == (3, model.config.hidden_size) and
                    torch.isfinite(previous["hidden"]).all()):
                completed.append(episode_id)
                continue
        frames = []
        try:
            frames = [image(args.records_root / args.part / record["initial_image"])] + [
                image(args.records_root / args.part / turn["image"])
                for turn in record["turns"]]
            midpoint = max(1, record["turn_count"] // 2)
            correct = [score(model, processor, frames, record["turns"],
                             record["instruction"], count,
                             tuple(action_tokens), format_text)
                       for count in (0, midpoint, record["turn_count"])]
            wrong = None
            if label["safe_wrong_instruction"]:
                wrong = score(model, processor, frames, record["turns"],
                              record["wrong_instruction"], record["turn_count"],
                              tuple(action_tokens), format_text)
            payload = {
                "schema": "stop_history_sft_features_v1",
                "episode_id": episode_id, "part": args.part,
                "manifest_sha256": manifest_sha,
                "record_sha256": record_sha,
                "model_config_sha256": model_hash,
                "prompt_sha256": prompt_hash,
                "prompt_version": PROMPT_VERSION,
                "hidden": torch.stack([row[0] for row in correct]),
                "stop_margin": torch.tensor([row[1] for row in correct]),
                "context_tokens": torch.tensor([row[2] for row in correct]),
                "wrong_hidden": wrong[0] if wrong is not None else torch.empty(0),
                "wrong_stop_margin": float(wrong[1]) if wrong is not None else float("nan"),
                "wrong_context_tokens": int(wrong[2]) if wrong is not None else 0,
            }
            temp = cache_path.with_suffix(".tmp")
            torch.save(payload, temp)
            os.replace(temp, cache_path)
            completed.append(episode_id)
            if number % 10 == 0:
                print(f"{args.part} states {number}/{len(selected)} elapsed_s={time.time()-begun:.1f}",
                      flush=True)
        except Exception as exc:
            errors.append({"episode_id": episode_id, "error": repr(exc)})
            print(f"ERROR {args.part}/{episode_id}: {exc!r}", flush=True)
        finally:
            for frame in frames:
                frame.close()
    summary = {"schema": "stop_history_sft_feature_cache_v1",
               "part": args.part, "requested": len(selected),
               "completed": len(completed),
               "completed_episode_ids": completed,
               "errors": errors,
               "manifest_sha256": manifest_sha,
               "model_config_sha256": model_hash,
               "prompt_sha256": prompt_hash,
               "prompt_version": PROMPT_VERSION,
               "limit": args.limit,
               "elapsed_seconds": time.time() - begun}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or len(completed) != len(selected):
        raise RuntimeError(f"incomplete {args.part} features: {len(completed)}/{len(selected)}")
    print(json.dumps({key: summary[key] for key in
                      ("part", "requested", "completed", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
