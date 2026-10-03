"""Cache frozen spatial SigLIP features for the ordered-clause experiment.

The source records and RGB files stay in the licensed experiment checkout.
The cache is keyed by the frozen clause manifest and model configuration.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "calibration"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or not 1 <= args.batch_size <= 32:
        raise ValueError("invalid shard or batch size")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["selected"][args.part]
    if manifest["schema"] != "clause_alignment_manifest_v1" or len(rows) != {
        "fit": 512, "calibration": 128}[args.part]:
        raise ValueError("unexpected clause source")
    manifest_sha = digest(args.manifest)
    model_sha = digest(args.model / "model.safetensors")
    config_sha = digest(args.model / "config.json")
    processor_sha = digest(args.model / "preprocessor_config.json")
    selected = rows[args.shard::args.shards]
    if not selected:
        raise ValueError("empty shard")

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                               use_fast=False)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).cuda().eval()
    output = args.output_root / args.part / "records"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frames = 0
    for i, row in enumerate(selected):
        episode_id = row["episode_id"]
        source = args.record_root / args.part / "records" / f"{episode_id}.json"
        if digest(source) != row["record_sha256"]:
            raise ValueError(f"source hash mismatch: {episode_id}")
        record = json.loads(source.read_text())
        if record["episode_id"] != episode_id or record["scene_id"] != row["scene_id"] or \
                len(record["frames"]) != 6:
            raise ValueError(f"source identity mismatch: {episode_id}")
        destination = output / f"{episode_id}.pt"
        if destination.exists():
            old = torch.load(destination, map_location="cpu", weights_only=True)
            if old["manifest_sha256"] == manifest_sha and old["record_sha256"] == row["record_sha256"] and \
                    old["model_sha256"] == model_sha and old["config_sha256"] == config_sha and \
                    old["processor_sha256"] == processor_sha and \
                    old["patches"].shape == (6, 49, 768) and \
                    bool(torch.isfinite(old["patches"]).all()):
                frames += 6
                continue
            raise ValueError(f"conflicting cached episode: {episode_id}")
        patches = []
        images = []
        try:
            for frame in record["frames"]:
                path = (args.record_root / args.part / frame["image"]).resolve()
                if not path.is_relative_to((args.record_root / args.part).resolve()):
                    raise ValueError("frame path escapes source root")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for offset in range(0, len(images), args.batch_size):
                    batch = processor(images=images[offset:offset + args.batch_size],
                                      return_tensors="pt")["pixel_values"].cuda()
                    tokens = model.vision_model(pixel_values=batch,
                                                 return_dict=True).last_hidden_state
                    if tokens.shape[1:] != (196, 768):
                        raise ValueError("unexpected SigLIP patch geometry")
                    pooled = F.avg_pool2d(tokens.float().reshape(-1, 14, 14, 768)
                                         .permute(0, 3, 1, 2), kernel_size=2, stride=2)
                    patches.append(pooled.permute(0, 2, 3, 1).reshape(-1, 49, 768).cpu().half())
        finally:
            for image in images:
                image.close()
        stacked = torch.cat(patches)
        if stacked.shape != (6, 49, 768) or not bool(torch.isfinite(stacked).all()):
            raise ValueError(f"invalid features: {episode_id}")
        payload = {"schema": "clause_spatial_siglip_v1", "episode_id": episode_id,
                   "scene_id": row["scene_id"], "manifest_sha256": manifest_sha,
                   "record_sha256": row["record_sha256"], "model_sha256": model_sha,
                   "config_sha256": config_sha, "processor_sha256": processor_sha,
                   "patch_geometry": "14x14 average pooled to 7x7; frozen vision last hidden state",
                   "patches": stacked}
        temp = destination.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, destination)
        frames += 6
        if (i + 1) % 32 == 0:
            print(f"{args.part} shard {args.shard}: {i + 1}/{len(selected)} "
                  f"in {time.time() - started:.1f}s", flush=True)
    summary = {"schema": "clause_spatial_cache_summary_v1", "part": args.part,
               "shard": args.shard, "shards": args.shards,
               "episodes": len(selected), "frames": frames,
               "manifest_sha256": manifest_sha, "model_sha256": model_sha,
               "config_sha256": config_sha, "processor_sha256": processor_sha,
               "elapsed_seconds": time.time() - started}
    path = args.output_root / args.part / f"summary_{args.shard}_of_{args.shards}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
