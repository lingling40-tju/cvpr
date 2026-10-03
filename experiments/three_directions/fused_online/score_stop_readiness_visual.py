"""Probe a frozen goal-grounded visual representation on on-policy STOPs.

This is a train-scene diagnostic. Fixed 64-token SigLIP padding and one
image/text pair per forward match the prior online reward service.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModel, AutoProcessor


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def similarity(model, processor, image_path, instruction):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((336, 336))
        image_input = processor(images=[image], return_tensors="pt")
    text_input = processor(text=[instruction], padding="max_length",
                           max_length=64, truncation=True, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        visual = model.get_image_features(
            pixel_values=image_input["pixel_values"].to("cuda:0"))
        language = model.get_text_features(
            **{key: value.to("cuda:0") for key, value in text_input.items()})
    return float((F.normalize(visual.float(), dim=-1) *
                  F.normalize(language.float(), dim=-1)).sum(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["schema"] == "stop_readiness_onpolicy_train_v1"
    source = json.loads((args.records_root / "summary.json").read_text())
    assert source["manifest_sha256"] == digest(args.manifest)
    assert source["completed_trajectories"] == 531 and not source["errors"]
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
    assert adapter["text_padding"] == "fixed_max_length_64"
    assert adapter["model_config_sha256"] == digest(args.model / "config.json")
    processor = AutoProcessor.from_pretrained(str(args.model),
                                              local_files_only=True,
                                              use_fast=False)
    base = AutoModel.from_pretrained(str(args.model), local_files_only=True,
                                     torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"])).cuda().eval()
    set_peft_model_state_dict(model, adapter["adapter"])
    records = manifest["records"][:args.limit_records or None]
    rows = []
    for index, plan in enumerate(records, 1):
        source = json.loads((args.records_root / "records" /
                             f"{plan['record_id']}.json").read_text())
        assert source["record_id"] == plan["record_id"]
        scores = [similarity(model, processor,
                             args.records_root / frame["image"],
                             source["instruction"])
                  for frame in source["frames"]]
        rows.append({"record_id": plan["record_id"],
                     "initial_score": scores[0],
                     "terminal_score": scores[1]})
        if index % 20 == 0 or index == len(records):
            print(f"visual {index}/{len(records)}", flush=True)
    report = {
        "schema": "stop_readiness_onpolicy_visual_score_v1",
        "interpretation": "Frozen train-scene on-policy diagnostic; no navigation result.",
        "manifest_sha256": digest(args.manifest),
        "adapter_sha256": digest(args.adapter),
        "model_config_sha256": digest(args.model / "config.json"),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
