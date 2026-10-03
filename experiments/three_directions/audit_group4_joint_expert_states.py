"""Independently verify complete cached same-start expert state coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from history_grounding_lora import digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--fit-shards", type=int, default=3)
    parser.add_argument("--development-shards", type=int, default=1)
    parser.add_argument("--audit-shards", type=int, default=1)
    parser.add_argument("--include-audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    source_id = digest(args.checkpoint)
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("wrong joint expert manifest")
    partitions = [("fit", args.fit_shards),
                  ("development", args.development_shards)]
    if args.include_audit:
        partitions.append(("audit", args.audit_shards))
    all_scenes = set()
    report = {"schema": "group4_joint_expert_state_audit_v1",
              "manifest_sha256": manifest_sha,
              "source_id": source_id, "parts": {}}
    for part, shards in partitions:
        rows = manifest["selected"][part]
        if shards < 1:
            raise ValueError("invalid shard count")
        for shard in range(shards):
            path = args.cache_root / part / f"summary_shard{shard}of{shards}.json"
            summary = json.loads(path.read_text())
            expected = rows[shard::shards]
            if summary["schema"] != "group4_joint_expert_state_shard_v1" or \
                    summary["part"] != part or summary["shard"] != shard or \
                    summary["shards"] != shards or \
                    summary["requested"] != len(expected) or \
                    summary["completed"] != len(expected) or \
                    summary["manifest_sha256"] != manifest_sha or \
                    summary["source_id"] != source_id:
                raise ValueError(f"incomplete expert state shard {part}/{shard}")
        scenes = set()
        for row in rows:
            eid = str(row["episode_id"])
            path = args.cache_root / part / "records" / f"{eid}.pt"
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache["schema"] != "group4_joint_expert_state_v1" or \
                    cache["episode_id"] != eid or cache["part"] != part or \
                    cache["manifest_sha256"] != manifest_sha or \
                    cache["source_id"] != source_id or \
                    cache["record_sha256"] != row["record_sha256"] or \
                    cache["correct"].shape != (2048,) or \
                    cache["wrong"].shape != (2048,) or \
                    not bool(torch.isfinite(cache["correct"]).all()) or \
                    not bool(torch.isfinite(cache["wrong"]).all()):
                raise ValueError(f"invalid expert state {part}/{eid}")
            scenes.add(row["scene_id"])
        if scenes & all_scenes or len(scenes) != manifest["inventory"][part]["scenes"]:
            raise ValueError(f"expert scene leakage or missing scene {part}")
        all_scenes.update(scenes)
        report["parts"][part] = {"expert_contrasts": len(rows),
                                 "states": 2 * len(rows),
                                 "scenes": len(scenes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
