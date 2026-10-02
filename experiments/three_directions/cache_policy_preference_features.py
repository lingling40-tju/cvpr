"""Cache frozen SigLIP features for replayed group-four train trajectories."""

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
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary = json.loads((args.records_root / "summary.json").read_text())
    if summary["pairs"] != len(manifest["pairs"]) or \
            summary["requested_trajectories"] != 2 * len(manifest["pairs"]) or \
            summary["completed_trajectories"] != 2 * len(manifest["pairs"]) or \
            summary["errors"] or \
            summary["manifest_sha256"] != digest(args.manifest):
        raise ValueError("policy frame collection incomplete")
    records = []
    for pair in manifest["pairs"]:
        for role in ("success", "failure"):
            record_id = pair["pair_id"] + "_" + role
            record = json.loads((args.records_root / "records" /
                                 f"{record_id}.json").read_text())
            if record["record_id"] != record_id or record["episode_id"] != pair["episode_id"] or \
                    record["scene_id"] != pair["scene_id"] or \
                    record["split"] != pair["split"] or record["role"] != role or \
                    len(record["frames"]) != 4 or \
                    record["frames"][0]["action_index"] != 0 or \
                    record["frames"][-1]["action_index"] != record["motion_actions"]:
                raise ValueError(f"bad policy frame record {record_id}")
            if abs(record["replayed_terminal_distance_m_for_audit_only"] -
                   pair[role]["terminal_distance_m_for_replay_audit_only"]) > 0.25:
                raise ValueError(f"replay drift {record_id}")
            records.append(record)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    texts = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = processor(text=[row["instruction"] for row in
                                    records[start:start + args.batch_size]],
                              return_tensors="pt", padding=True, truncation=True)
            features = model.get_text_features(**{key: value.to(device)
                                                 for key, value in batch.items()})
            texts.append(F.normalize(features.float(), dim=-1).cpu())
    paths = [args.records_root / frame["image"]
             for row in records for frame in row["frames"]]
    images = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            opened = []
            try:
                for path in paths[start:start + args.batch_size]:
                    opened.append(Image.open(path).convert("RGB"))
                batch = processor(images=opened, return_tensors="pt")
                features = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
                images.append(F.normalize(features.float(), dim=-1).cpu())
            finally:
                for frame in opened:
                    frame.close()
            if (start // args.batch_size) % 20 == 0:
                print(f"{min(start + args.batch_size, len(paths))}/{len(paths)} frames", flush=True)
    visual = torch.cat(images).reshape(len(records), 4, -1)
    language = torch.cat(texts)
    if not torch.isfinite(visual).all() or not torch.isfinite(language).all():
        raise ValueError("nonfinite policy features")
    payload = {"schema": "policy_preference_features_v1",
               "manifest_sha256": digest(args.manifest),
               "backbone": args.model.name,
               "model_config_sha256": digest(args.model / "config.json"),
               "record_ids": [row["record_id"] for row in records],
               "pair_ids": [row["pair_id"] for row in records],
               "roles": [row["role"] for row in records],
               "splits": [row["split"] for row in records],
               "scene_ids": [row["scene_id"] for row in records],
               "episode_ids": torch.tensor([row["episode_id"] for row in records],
                                           dtype=torch.int64),
               "action_indices": torch.tensor([[frame["action_index"] for frame in row["frames"]]
                                               for row in records], dtype=torch.int64),
               "texts": language, "images": visual}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"trajectories": len(records), "frames": len(paths),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
