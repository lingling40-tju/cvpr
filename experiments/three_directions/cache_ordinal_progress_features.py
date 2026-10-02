"""Cache frozen image/text embeddings for train-only expert frame records.

This is an offline feature pass; it does not train a reward model or score any
val-unseen episode. Use the same frozen backbone for fit and calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ordinal_progress_selection import select_episode_ids


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "calibration"), required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    episode_ids = select_episode_ids(manifest, args.subset, args.limit)
    base = args.records_root / args.subset
    records = []
    for episode_id in episode_ids:
        record = json.loads((base / "records" / f"{episode_id}.json").read_text())
        if record["episode_id"] != episode_id or len(record["frames"]) < 2:
            raise ValueError(f"bad frame record for {episode_id}")
        records.append(record)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

    texts = [item["instruction"] for item in records]
    text_features = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = processor(text=texts[start:start + args.batch_size],
                              return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in batch.items()}
            features = model.get_text_features(**inputs)
            text_features.append(F.normalize(features.float(), dim=-1).cpu())
    text_features = torch.cat(text_features)

    frame_rows = [(record["episode_id"], offset, frame["progress_fraction"],
                   base / frame["image"])
                  for record in records for offset, frame in enumerate(record["frames"])]
    image_features = []
    with torch.inference_mode():
        for start in range(0, len(frame_rows), args.batch_size):
            paths = [item[3] for item in frame_rows[start:start + args.batch_size]]
            images = []
            try:
                for path in paths:
                    images.append(Image.open(path).convert("RGB"))
                batch = processor(images=images, return_tensors="pt")
                features = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
                image_features.append(F.normalize(features.float(), dim=-1).cpu())
            finally:
                for image in images:
                    image.close()
            if (start // args.batch_size) % 20 == 0:
                print(f"{args.subset}: {min(start + args.batch_size, len(frame_rows))}/{len(frame_rows)} frames", flush=True)
    image_features = torch.cat(image_features)
    if not torch.isfinite(text_features).all() or not torch.isfinite(image_features).all():
        raise ValueError("nonfinite visual/text features")
    payload = {
        "subset": args.subset,
        "backbone": args.model.name,
        "manifest_sha256": digest(args.manifest),
        "model_config_sha256": digest(args.model / "config.json"),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.int64),
        "texts": text_features,
        "frame_episode_ids": torch.tensor([row[0] for row in frame_rows], dtype=torch.int64),
        "frame_offsets": torch.tensor([row[1] for row in frame_rows], dtype=torch.int64),
        "progress_fractions": torch.tensor([row[2] for row in frame_rows], dtype=torch.float32),
        "images": image_features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "episodes": len(records),
                      "frames": len(frame_rows), "feature_dim": image_features.shape[-1],
                      "subset": args.subset, "backbone": payload["backbone"],
                      "manifest_sha256": payload["manifest_sha256"]}))


if __name__ == "__main__":
    main()
