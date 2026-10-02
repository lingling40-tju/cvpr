"""Cheap text-only preflight for explicit goal conditioning.

Frozen SigLIP visual frames are reused. Only wrong-instruction text vectors
are newly encoded. This is an exploratory development diagnostic, not a
navigation metric or a predeclared acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--siglip-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    frozen = torch.load(args.siglip_features, map_location="cpu", weights_only=False)
    if v2["source_manifest_sha256"] != digest(args.source_manifest) or \
            frozen["manifest_sha256"] != digest(args.source_manifest) or \
            frozen["images"].shape != (2 * len(source["pairs"]), 4, 768) or \
            frozen["texts"].shape != (2 * len(source["pairs"]), 768):
        raise ValueError("frozen SigLIP coverage mismatch")
    by_id = {row["pair_id"]: row for row in v2["pairs"]}
    indices = [i for i, pair in enumerate(source["pairs"])
               if by_id[pair["pair_id"]]["split"] == "development"]
    if len(indices) != v2["counts"]["development"]:
        raise ValueError("development split mismatch")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).cuda().eval()
    wrong = []
    texts = [by_id[source["pairs"][i]["pair_id"]]["swapped_instruction"]
             for i in indices]
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = processor(text=texts[start:start + 32], padding=True,
                              truncation=True, return_tensors="pt")
            encoded = model.get_text_features(**{key: value.cuda()
                                                 for key, value in batch.items()})
            wrong.append(F.normalize(encoded.float(), dim=-1).cpu())
    wrong = torch.cat(wrong)
    images = frozen["images"].float()
    original = frozen["texts"].float()
    results = {}
    for name, visual in (
        ("endpoint", images[2 * torch.tensor(indices), -1]),
        ("last_two_max", images[2 * torch.tensor(indices), -2:].mean(dim=1)),
        ("endpoint_minus_start", images[2 * torch.tensor(indices), -1] -
         images[2 * torch.tensor(indices), 0]),
    ):
        correct = (visual * original[2 * torch.tensor(indices)]).sum(-1)
        mismatched = (visual * wrong).sum(-1)
        results[name] = {"correct_over_wrong": float((correct > mismatched).float().mean()),
                         "mean_margin": float((correct - mismatched).mean())}
    report = {"schema": "explicit_goal_alignment_preflight_v1",
              "split": "development", "pairs": len(indices),
              "interpretation": "Frozen text/visual preflight only; no learned reward or RL.",
              "metrics": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
