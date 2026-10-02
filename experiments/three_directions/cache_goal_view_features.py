"""Embed four train-only goal-pose views with the frozen SigLIP backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "calibration"), required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--expert-features", type=Path,
                        help="Reuse text embeddings when episode IDs exactly match")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ids = manifest["subsets"][args.subset]["episode_ids"]
    expert = None
    if args.expert_features:
        expert = torch.load(args.expert_features, map_location="cpu", weights_only=False)
        if expert["subset"] != args.subset or expert["episode_ids"].tolist() != ids or \
                expert["backbone"] != args.model.name:
            raise ValueError("source features mismatch")
    records = []
    for eid in ids:
        record = json.loads((args.records_root / args.subset / "records" /
                             f"{eid}.json").read_text())
        if record["episode_id"] != eid or len(record["views"]) != 4:
            raise ValueError(f"goal record mismatch for {eid}")
        records.append(record)
    if expert and [item["scene_id"] for item in records] != expert["scene_ids"]:
        raise ValueError("scene mismatch against expert features")
    rows = [(args.records_root / args.subset / view["image"])
            for record in records for view in record["views"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    if expert:
        text_features = expert["texts"]
    else:
        parts = []
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                texts = [item["instruction"] for item in records[start:start + args.batch_size]]
                batch = processor(text=texts, return_tensors="pt", padding=True,
                                  truncation=True)
                features = model.get_text_features(**{key: value.to(device)
                                                     for key, value in batch.items()})
                parts.append(F.normalize(features.float(), dim=-1).cpu())
        text_features = torch.cat(parts)
    features = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            images = []
            try:
                for path in rows[start:start + args.batch_size]:
                    images.append(Image.open(path).convert("RGB"))
                batch = processor(images=images, return_tensors="pt")
                emb = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
                features.append(F.normalize(emb.float(), dim=-1).cpu())
            finally:
                for image in images:
                    image.close()
            if (start // args.batch_size) % 20 == 0:
                print(f"{args.subset} {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    visual = torch.cat(features).reshape(len(ids), 4, -1)
    if not torch.isfinite(visual).all():
        raise ValueError("nonfinite goal embeddings")
    payload = {"subset": args.subset,
               "episode_ids": torch.tensor(ids, dtype=torch.int64),
               "scene_ids": [item["scene_id"] for item in records],
               "texts": text_features, "images": visual,
               "backbone": args.model.name,
               "manifest_sha256": digest(args.manifest),
               "model_config_sha256": digest(args.model / "config.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"episodes": len(ids), "views": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
