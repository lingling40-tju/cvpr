"""Audit frozen SigLIP local-region features before any scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepare_clause_alignment_manifest import digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "clause_alignment_manifest_v1":
        raise ValueError("unexpected source")
    hashes = {"manifest_sha256": digest(args.manifest),
              "model_sha256": digest(args.model / "model.safetensors"),
              "config_sha256": digest(args.model / "config.json"),
              "processor_sha256": digest(args.model / "preprocessor_config.json")}
    partitions = {}
    scene_sets = []
    for part in ("fit", "calibration"):
        rows = manifest["selected"][part]
        expected_ids = {row["episode_id"] for row in rows}
        files = list((args.cache_root / part / "records").glob("*.pt"))
        if len(files) != len(expected_ids) or {int(p.stem) for p in files} != expected_ids:
            raise ValueError(f"episode coverage mismatch: {part}")
        scenes = set()
        for row in rows:
            payload = torch.load(args.cache_root / part / "records" /
                                 f"{row['episode_id']}.pt", map_location="cpu",
                                 weights_only=True)
            features = payload["features"].float()
            if payload["schema"] != "clause_region_siglip_v1" or \
                    payload["episode_id"] != row["episode_id"] or \
                    payload["scene_id"] != row["scene_id"] or \
                    payload["record_sha256"] != row["record_sha256"] or \
                    any(payload[key] != value for key, value in hashes.items()) or \
                    tuple(payload["regions"]) != ("full", "top_left", "top_right",
                                                   "bottom_left", "bottom_right") or \
                    features.shape != (6, 5, 768) or \
                    not bool(torch.isfinite(features).all()) or \
                    not bool(((features.norm(dim=-1) - 1).abs() < 0.002).all()):
                raise ValueError(f"invalid region cache {part}/{row['episode_id']}")
            scenes.add(row["scene_id"])
        scene_sets.append(scenes)
        partitions[part] = {"episodes": len(rows), "frames": 6 * len(rows),
                            "pooled_region_vectors": 30 * len(rows),
                            "scenes": len(scenes)}
    if scene_sets[0] & scene_sets[1]:
        raise ValueError("scene overlap")
    report = {"schema": "clause_region_cache_audit_v1", "source_hashes": hashes,
              "partitions": partitions, "scene_disjoint": True,
              "interpretation": "frozen pretrained local-region coverage only; no score or navigation result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
