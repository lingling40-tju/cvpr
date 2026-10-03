"""Encode fixed local crops through SigLIP's pretrained image pooler.

Only R2R-train replay frames are read. Source images remain remote.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from prepare_clause_alignment_manifest import digest


REGIONS = ("full", "top_left", "top_right", "bottom_left", "bottom_right")


def crops(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    crop_width, crop_height = 2 * width // 3, 2 * height // 3
    if min(crop_width, crop_height) < 32:
        raise ValueError("source frame too small")
    regions = [image.copy()]
    for top in (0, height - crop_height):
        for left in (0, width - crop_width):
            regions.append(image.crop((left, top, left + crop_width,
                                       top + crop_height)))
    if len(regions) != len(REGIONS):
        raise ValueError("crop geometry changed")
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "calibration"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or not 1 <= args.batch_size <= 32:
        raise ValueError("invalid shard or batch size")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["selected"][args.part]
    if manifest["schema"] != "clause_alignment_manifest_v1" or len(rows) != {
        "fit": 512, "calibration": 128}[args.part]:
        raise ValueError("unexpected clause source")
    hashes = {"manifest_sha256": digest(args.manifest),
              "model_sha256": digest(args.model / "model.safetensors"),
              "config_sha256": digest(args.model / "config.json"),
              "processor_sha256": digest(args.model / "preprocessor_config.json")}
    selected = rows[args.shard::args.shards]
    if not selected:
        raise ValueError("empty shard")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                               use_fast=False)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).cuda().eval()
    destination = args.output_root / args.part / "records"
    destination.mkdir(parents=True, exist_ok=True)
    part_root = (args.record_root / args.part).resolve()
    started = time.time()
    frames = 0
    for i, row in enumerate(selected):
        episode_id = row["episode_id"]
        source = part_root / "records" / f"{episode_id}.json"
        if digest(source) != row["record_sha256"]:
            raise ValueError(f"source hash mismatch {episode_id}")
        record = json.loads(source.read_text())
        if record["episode_id"] != episode_id or record["scene_id"] != row["scene_id"] or \
                len(record["frames"]) != 6:
            raise ValueError(f"source identity mismatch {episode_id}")
        path = destination / f"{episode_id}.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            if old["schema"] == "clause_region_siglip_v1" and \
                    old["episode_id"] == episode_id and \
                    old["record_sha256"] == row["record_sha256"] and \
                    all(old[key] == value for key, value in hashes.items()) and \
                    old["features"].shape == (6, 5, 768) and \
                    bool(torch.isfinite(old["features"]).all()):
                frames += 6
                continue
            raise ValueError(f"conflicting cached episode {episode_id}")
        regions = []
        try:
            for frame in record["frames"]:
                frame_path = (part_root / frame["image"]).resolve()
                if not frame_path.is_relative_to(part_root):
                    raise ValueError("frame escapes source root")
                with Image.open(frame_path) as source_image:
                    rgb = source_image.convert("RGB")
                regions.extend(crops(rgb))
                rgb.close()
            parts = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(regions), args.batch_size):
                    pixels = processor(images=regions[start:start + args.batch_size],
                                       return_tensors="pt")["pixel_values"].cuda()
                    features = model.get_image_features(pixel_values=pixels)
                    parts.append(F.normalize(features.float(), dim=-1).cpu().half())
            stacked = torch.cat(parts).reshape(6, 5, 768)
        finally:
            for image in regions:
                image.close()
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError(f"nonfinite region features {episode_id}")
        payload = {"schema": "clause_region_siglip_v1", "episode_id": episode_id,
                   "scene_id": row["scene_id"], "record_sha256": row["record_sha256"],
                   **hashes, "regions": REGIONS,
                   "crop_geometry": "full plus four corner-anchored 2/3 width x 2/3 height overlapping crops",
                   "features": stacked}
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)
        frames += 6
        if (i + 1) % 32 == 0:
            print(f"{args.part} shard {args.shard}: {i + 1}/{len(selected)} "
                  f"in {time.time() - started:.1f}s", flush=True)
    summary = {"schema": "clause_region_cache_summary_v1", "part": args.part,
               "shard": args.shard, "shards": args.shards,
               "episodes": len(selected), "frames": frames,
               "regions": len(REGIONS), **hashes,
               "elapsed_seconds": time.time() - started}
    summary_path = args.output_root / args.part / f"summary_{args.shard}_of_{args.shards}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
