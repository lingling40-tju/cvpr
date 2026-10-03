"""Verify exact cached-state coverage for the frozen group-four screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from history_grounding_lora import digest, rid


ANCHORS = (3, 6, 9, 12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--include-audit", action="store_true")
    parser.add_argument("--audit-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit_shards < 1 or (not args.include_audit and args.audit_shards != 1):
        raise ValueError("invalid audit shard count")
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    report = {"schema": "group_relative_state_cache_audit_v1",
              "manifest_sha256": manifest_sha, "source_id": args.source_id,
              "parts": {}}
    all_scenes = set()
    partitions = [("fit", 3), ("development", 1)]
    if args.include_audit:
        partitions.append(("audit", args.audit_shards))
    for part, shards in partitions:
        plans = sorted(manifest["selected"][part], key=rid)
        for shard in range(shards):
            name = f"summary_shard{shard}of{shards}.json"
            summary = json.loads((args.cache_root / part / name).read_text())
            expected = plans[shard::shards]
            if (summary["part"] != part or summary["shard"] != shard or
                    summary["shards"] != shards or
                    summary["requested"] != len(expected) or
                    summary["completed"] != len(expected) or
                    summary["manifest_sha256"] != manifest_sha or
                    summary["source_id"] != args.source_id):
                raise ValueError(f"incomplete cache shard {part}/{shard}")
        count = 0
        scenes = set()
        for plan in plans:
            record_id = rid(plan)
            record = json.loads((args.turn_root / part / "records" /
                                 f"{record_id}.json").read_text())
            expected = [turn["original_turn_index"] for turn in record["turns"]
                        if turn["original_turn_index"] in ANCHORS]
            cache = torch.load(args.cache_root / part / "records" /
                               f"{record_id}.pt", map_location="cpu",
                               weights_only=True)
            if (cache["schema"] != "group_relative_state_cache_v1" or
                    cache["record_id"] != record_id or
                    cache["source_id"] != args.source_id or
                    cache["manifest_sha256"] != manifest_sha or
                    cache["anchor_turns"] != expected or
                    cache["hidden"].shape != (len(expected), 2048) or
                    not torch.isfinite(cache["hidden"]).all()):
                raise ValueError(f"invalid cached state {part}/{record_id}")
            count += len(expected)
            scenes.add(plan["scene_id"])
        if scenes & all_scenes or len(scenes) != manifest["inventory"][part]["scenes"]:
            raise ValueError("scene leakage or missing scene")
        all_scenes.update(scenes)
        report["parts"][part] = {"trajectories": len(plans),
                                 "states": count, "scenes": len(scenes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
