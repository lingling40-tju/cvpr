"""Independently audit frozen spatial-feature coverage and provenance."""

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
    parser.add_argument("--fit-text", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "clause_alignment_manifest_v1":
        raise ValueError("unexpected manifest")
    hashes = {"manifest_sha256": digest(args.manifest),
              "model_sha256": digest(args.model / "model.safetensors"),
              "config_sha256": digest(args.model / "config.json"),
              "processor_sha256": digest(args.model / "preprocessor_config.json")}
    partitions = {}
    all_scenes = []
    for part, text_path in (("fit", args.fit_text),
                            ("calibration", args.calibration_text)):
        rows = manifest["selected"][part]
        expected_ids = {row["episode_id"] for row in rows}
        actual_files = list((args.cache_root / part / "records").glob("*.pt"))
        if {int(p.stem) for p in actual_files} != expected_ids or \
                len(actual_files) != len(expected_ids):
            raise ValueError(f"coverage mismatch: {part}")
        text = torch.load(text_path, map_location="cpu", weights_only=True)
        if text["part"] != part or text["manifest_sha256"] != hashes["manifest_sha256"] or \
                text["model_config_sha256"] != hashes["config_sha256"] or \
                text["episode_ids"].tolist() != [r["episode_id"] for r in rows]:
            raise ValueError(f"text/spatial source mismatch: {part}")
        scenes = set()
        for row in rows:
            payload = torch.load(args.cache_root / part / "records" /
                                 f"{row['episode_id']}.pt", map_location="cpu",
                                 weights_only=True)
            if payload["schema"] != "clause_spatial_siglip_v1" or \
                    payload["episode_id"] != row["episode_id"] or \
                    payload["scene_id"] != row["scene_id"] or \
                    payload["record_sha256"] != row["record_sha256"] or \
                    any(payload[key] != value for key, value in hashes.items()) or \
                    payload["patches"].shape != (6, 49, 768) or \
                    payload["patches"].dtype != torch.float16 or \
                    not bool(torch.isfinite(payload["patches"]).all()):
                raise ValueError(f"invalid cache: {part}/{row['episode_id']}")
            scenes.add(row["scene_id"])
        all_scenes.append(scenes)
        partitions[part] = {"episodes": len(rows), "frames": len(rows) * 6,
                            "spatial_tokens": len(rows) * 6 * 49,
                            "scenes": len(scenes),
                            "text_sha256": digest(text_path)}
    if all_scenes[0] & all_scenes[1]:
        raise ValueError("fit/calibration scene overlap")
    result = {"schema": "clause_spatial_cache_audit_v1",
              "source_hashes": hashes, "partitions": partitions,
              "scene_disjoint": True,
              "interpretation": "frozen feature coverage only; no learned reward or navigation result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
