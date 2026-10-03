"""Encode deterministic instruction clauses with fixed-length SigLIP text input.

Only R2R-train instructions are read. The public result contains embeddings
and hashes, never source instruction strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from prepare_clause_alignment_manifest import clauses, digest, text_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "clause_alignment_manifest_v1" or \
            not 1 <= args.batch_size <= 128:
        raise ValueError("bad clause source or batch size")
    rows = manifest["selected"][args.part]
    if len(rows) != {"fit": 512, "calibration": 128}[args.part]:
        raise ValueError("clause coverage changed")
    strings = []
    offsets = [0]
    full = []
    for row in rows:
        eid = row["episode_id"]
        path = args.record_root / args.part / "records" / f"{eid}.json"
        if digest(path) != row["record_sha256"]:
            raise ValueError(f"record changed {eid}")
        record = json.loads(path.read_text())
        parts = clauses(record["instruction"])
        if record["episode_id"] != eid or \
                record["scene_id"] != row["scene_id"] or \
                text_hash(record["instruction"]) != row["instruction_sha256"] or \
                [text_hash(piece) for piece in parts] != row["clause_sha256"]:
            raise ValueError(f"clause mismatch {eid}")
        strings.extend(parts)
        offsets.append(len(strings))
        full.append(record["instruction"])
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).cuda().eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

    def encode(texts: list[str]) -> torch.Tensor:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                batch = processor(text=texts[start:start + args.batch_size],
                                  return_tensors="pt", padding="max_length",
                                  max_length=64, truncation=True)
                if batch["input_ids"].shape[1] != 64:
                    raise ValueError("SigLIP text padding changed")
                inputs = {k: v.cuda() for k, v in batch.items()}
                features = model.get_text_features(**inputs)
                chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks).to(torch.float16)

    clause_features = encode(strings)
    full_features = encode(full)
    if not bool(torch.isfinite(clause_features).all()) or \
            not bool(torch.isfinite(full_features).all()) or \
            clause_features.shape[-1] != full_features.shape[-1] or \
            len(offsets) != len(rows) + 1:
        raise ValueError("bad text feature cache")
    payload = {
        "schema": "clause_siglip_text_features_v1",
        "part": args.part, "manifest_sha256": digest(args.manifest),
        "model_config_sha256": digest(args.model / "config.json"),
        "text_padding_length": 64,
        "episode_ids": torch.tensor([r["episode_id"] for r in rows], dtype=torch.int64),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "clauses": clause_features, "full": full_features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"part": args.part, "episodes": len(rows),
                      "clauses": len(strings), "feature_dim": clause_features.shape[-1],
                      "manifest_sha256": payload["manifest_sha256"],
                      "model_config_sha256": payload["model_config_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
