"""Cache frozen SigLIP patch and instruction features from verified RGB replay.

Only fit/development groups are available. The simulator's geodesic
distance never enters visual/text encoding. Fixed 64-token text padding
matches SigLIP's final-position pooling at deployment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

from history_grounding_lora import digest, rid


ANCHORS = (3, 6, 9, 12)


def encode_texts(model, processor, texts: list[str], batch_size: int) -> dict:
    vectors = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = processor(text=chunk, padding="max_length",
                               max_length=64, truncation=True,
                               return_tensors="pt")
            output = model.get_text_features(**{key: value.cuda() for key, value
                                                in inputs.items()})
            output = F.normalize(output.float(), dim=-1).cpu().half()
            if not torch.isfinite(output).all() or output.shape != (len(chunk), 768):
                raise ValueError("invalid text embedding")
            vectors.update(zip(chunk, output))
    return vectors


def encode_images(model, processor, paths: list[Path],
                  batch_size: int) -> torch.Tensor:
    parts = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(paths), batch_size):
            opened = []
            try:
                for path in paths[start:start + batch_size]:
                    with Image.open(path) as source:
                        opened.append(source.convert("RGB"))
                pixels = processor(images=opened, return_tensors="pt")[
                    "pixel_values"].cuda()
                output = model.vision_model(pixel_values=pixels,
                                            return_dict=True).last_hidden_state
                output = output.float().cpu().half()
                if output.shape != (len(opened), 196, 768) or \
                        not torch.isfinite(output).all():
                    raise ValueError("invalid visual patch embedding")
                parts.append(output)
            finally:
                for image in opened:
                    image.close()
    return torch.cat(parts) if parts else \
           torch.empty((0, 196, 768), dtype=torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or not 1 <= args.batch_size <= 64:
        raise ValueError("invalid shard or batch size")
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.collection_audit.read_text())
    manifest_sha = digest(args.manifest)
    if (manifest["schema"] != "policy_group_relative_manifest_v1" or
            audit["manifest_sha256"] != manifest_sha or
            not audit["group_comparison_sample_gate"]["passed"]):
        raise ValueError("source replay not audited")
    source_id = digest(args.model / "model.safetensors")
    config_sha = digest(args.model / "config.json")
    processor_sha = digest(args.model / "preprocessor_config.json")
    plans = sorted(manifest["selected"][args.part], key=rid)
    plans = plans[args.shard::args.shards]
    if not plans:
        raise ValueError("empty shard")
    records = []
    root = args.turn_root / args.part
    for plan in plans:
        record_id = rid(plan)
        record = json.loads((root / "records" / f"{record_id}.json").read_text())
        if (record["record_id"] != record_id or
                record["manifest_sha256"] != manifest_sha or
                record["scene_id"] != plan["scene_id"]):
            raise ValueError(f"source record mismatch {record_id}")
        records.append(record)
    processor = AutoProcessor.from_pretrained(str(args.model),
                                              local_files_only=True,
                                              use_fast=False)
    model = AutoModel.from_pretrained(str(args.model),
                                      local_files_only=True).cuda().eval()
    texts = sorted({record["instruction"] for record in records})
    text_vectors = encode_texts(model, processor, texts, args.batch_size)
    output = args.output_root / args.part
    (output / "records").mkdir(parents=True, exist_ok=True)
    completed = 0
    frames = 0
    started = time.time()
    for start in range(0, len(records), 16):
        chunk = records[start:start + 16]
        pending = []
        for record in chunk:
            record_id = record["record_id"]
            turns = [turn for turn in record["turns"]
                     if turn["original_turn_index"] in ANCHORS]
            anchor_turns = [turn["original_turn_index"] for turn in turns]
            path = output / "records" / f"{record_id}.pt"
            if path.is_file():
                prior = torch.load(path, map_location="cpu", weights_only=True)
                if (prior["schema"] == "group_visual_tokens_v1" and
                        prior["record_id"] == record_id and
                        prior["manifest_sha256"] == manifest_sha and
                        prior["source_id"] == source_id and
                        prior["config_sha256"] == config_sha and
                        prior["processor_sha256"] == processor_sha and
                        prior["anchor_turns"] == anchor_turns and
                        prior["patches"].shape == (len(turns), 196, 768) and
                        prior["text"].shape == (768,) and
                        torch.isfinite(prior["patches"]).all() and
                        torch.isfinite(prior["text"]).all()):
                    completed += 1
                    frames += len(turns)
                    continue
            pending.append((record, turns, path))
        paths = [root / turn["image"] for _, turns, _ in pending for turn in turns]
        encoded = encode_images(model, processor, paths, args.batch_size)
        offset = 0
        for record, turns, path in pending:
            count = len(turns)
            patches = encoded[offset:offset + count].clone()
            offset += count
            payload = {"schema": "group_visual_tokens_v1",
                       "record_id": record["record_id"],
                       "manifest_sha256": manifest_sha,
                       "source_id": source_id,
                       "config_sha256": config_sha,
                       "processor_sha256": processor_sha,
                       "anchor_turns": [turn["original_turn_index"]
                                        for turn in turns],
                       "patches": patches,
                       "text": text_vectors[record["instruction"]]}
            temp = path.with_suffix(".tmp")
            torch.save(payload, temp)
            os.replace(temp, path)
            completed += 1
            frames += count
        if offset != len(encoded):
            raise ValueError("visual frame aggregation mismatch")
        print(f"{args.part} shard={args.shard}/{args.shards} "
              f"{min(start + len(chunk), len(records))}/{len(records)} "
              f"frames={frames} elapsed_s={time.time()-started:.1f}", flush=True)
    summary = {"schema": "group_visual_token_collection_v1",
               "part": args.part, "shard": args.shard, "shards": args.shards,
               "requested": len(records), "completed": completed,
               "frames": frames, "source_id": source_id,
               "config_sha256": config_sha,
               "processor_sha256": processor_sha,
               "manifest_sha256": manifest_sha,
               "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
