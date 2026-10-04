"""Format-only smoke for the published Route2Step instruction-analysis model.

This reads two SHA-selected R2R-train fit trajectories. It records text and
prompt provenance only; it does not score development examples, fit a reward,
or evaluate navigation. The prompt and frame sampling mirror the upstream
Route2Step MIA agent at its pinned model revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
import torch
from transformers import (AutoProcessor, Qwen2_5_VLConfig,
                          Qwen2_5_VLForConditionalGeneration)


SYSTEM = "You are an intelligent navigation robot."
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def selected_frames(record: dict, root: Path, anchor: int) -> tuple[list[Image.Image], int, int]:
    paths = [record["initial_image"]] + [turn["image"]
                                          for turn in record["turns"][:anchor]]
    indices = list(range(len(paths)))
    current = indices[-3:]
    history_end = max(0, len(indices) - 3 - 2)
    history = indices[:history_end]
    if len(history) > 13:
        raise ValueError("smoke anchor unexpectedly exceeds history budget")
    chosen = history + current
    images = []
    for index in chosen:
        with Image.open(root / paths[index]) as raw:
            images.append(raw.convert("RGB").resize((640, 480)))
    return images, len(history), len(current)


def prompt(instruction: str, history_count: int, current_count: int) -> str:
    total = history_count + current_count
    desc = f"Above are {total} images."
    if history_count:
        desc += (f" The first {history_count} images are the History trajectory,"
                 f" and the last {current_count} images are the Current view.")
    else:
        desc += " All of them are the Current view."
    return (
        f"{'<image>' * total}\n{desc}\n"
        f"Global Instruction: {instruction}\n"
        "Task: Analyze the history and current view to determine the current progress within the global "
        "instruction. Provide a structured report with the following format: <think> Current Instruction: "
        "<instruction> | Status: <Executing/Completed> | Next Instruction: <instruction> or None </think>\n"
        "<answer> Next Instruction to Execute </answer>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("model", "manifest", "record-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "policy_process_train_manifest_v1":
        raise ValueError("unexpected policy process manifest")
    plans = manifest["selected"]["fit"]
    chosen = sorted(plans, key=lambda row: hashlib.sha256(
        f"route2step-smoke/{row['seed']}/{row['episode_id']}/{row['variant']}".encode()
    ).digest())[:2]
    processor = AutoProcessor.from_pretrained(str(args.model),
                                              local_files_only=True,
                                              use_fast=False)
    # The public checkpoint exports both the legacy top-level Qwen2.5-VL
    # text fields and a newer nested text_config. Transformers 4.51.3 in the
    # validated ActiveVLN environment treats that nested dict as a model
    # config object and crashes. Its overlapping architecture fields agree,
    # so use the unchanged top-level fields without editing downloaded files.
    config_data = json.loads((args.model / "config.json").read_text())
    nested = config_data.pop("text_config")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "intermediate_size", "vocab_size"):
        if config_data[key] != nested[key]:
            raise ValueError(f"checkpoint text architecture mismatch: {key}")
    config_data["tie_word_embeddings"] = nested["tie_word_embeddings"]
    config = Qwen2_5_VLConfig(**config_data)
    model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise ValueError(f"checkpoint weights did not load exactly: {loading}")
    model = model.cuda().eval()
    output = []
    for plan in chosen:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        record_path = args.record_root / "fit" / "records" / f"{rid}.json"
        record = json.loads(record_path.read_text())
        if record["record_id"] != rid or \
                record["manifest_sha256"] != digest(args.manifest) or \
                len(record["turns"]) < 6:
            raise ValueError(f"invalid fit record {rid}")
        for anchor in (3, 6):
            images, num_h, num_c = selected_frames(
                record, args.record_root / "fit", anchor)
            try:
                query = prompt(record["instruction"], num_h, num_c)
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": query}]
                rendered = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                rendered = rendered.replace("<image>", IMAGE_TOKEN)
                inputs = processor(text=[rendered], images=images,
                                   return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=256,
                                               do_sample=False)
                response = processor.batch_decode(
                    generated[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0]
                output.append({
                    "record_id": rid, "record_sha256": digest(record_path),
                    "anchor": anchor, "history_images": num_h,
                    "current_images": num_c,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "response": response,
                    "has_answer_tag": bool(re.search(r"<answer>.*?</answer>",
                                                     response, re.S | re.I)),
                })
            finally:
                for image in images:
                    image.close()
    report = {
        "schema": "route2step_mia_format_smoke_v1",
        "model_config_sha256": digest(args.model / "config.json"),
        "manifest_sha256": digest(args.manifest),
        "upstream_prompt": "https://raw.githubusercontent.com/BUAA-GAMMA-LAB/Route2Step/refs/heads/main/agent_dual_qwen2_5_lm.py",
        "rows": output,
        "interpretation": "Format-only R2R-train fit smoke; no development score or navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"records": len(chosen), "queries": len(output),
                      "answer_tags": sum(row["has_answer_tag"] for row in output)}))


if __name__ == "__main__":
    main()
