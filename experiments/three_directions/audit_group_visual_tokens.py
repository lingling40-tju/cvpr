"""Verify exact SigLIP token cache coverage and group instruction identity."""

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
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--processor-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    report = {"schema": "group_visual_token_cache_audit_v1",
              "manifest_sha256": manifest_sha,
              "source_id": args.source_id,
              "config_sha256": args.config_sha,
              "processor_sha256": args.processor_sha,
              "parts": {}}
    all_scenes = set()
    for part, shards in (("fit", 3), ("development", 1)):
        plans = sorted(manifest["selected"][part], key=rid)
        frames = 0
        scenes = set()
        text_by_group = {}
        for shard in range(shards):
            path = args.cache_root / part / f"summary_shard{shard}of{shards}.json"
            summary = json.loads(path.read_text())
            count = len(plans[shard::shards])
            if (summary["part"] != part or summary["shard"] != shard or
                    summary["shards"] != shards or
                    summary["requested"] != count or
                    summary["completed"] != count or
                    summary["source_id"] != args.source_id or
                    summary["config_sha256"] != args.config_sha or
                    summary["processor_sha256"] != args.processor_sha or
                    summary["manifest_sha256"] != manifest_sha):
                raise ValueError(f"incomplete visual shard {part}/{shard}")
        for plan in plans:
            record_id = rid(plan)
            record = json.loads((args.turn_root / part / "records" /
                                 f"{record_id}.json").read_text())
            expected = [turn["original_turn_index"] for turn in record["turns"]
                        if turn["original_turn_index"] in ANCHORS]
            row = torch.load(args.cache_root / part / "records" /
                             f"{record_id}.pt", map_location="cpu",
                             weights_only=True)
            if (row["schema"] != "group_visual_tokens_v1" or
                    row["record_id"] != record_id or
                    row["manifest_sha256"] != manifest_sha or
                    row["source_id"] != args.source_id or
                    row["config_sha256"] != args.config_sha or
                    row["processor_sha256"] != args.processor_sha or
                    row["anchor_turns"] != expected or
                    row["patches"].shape != (len(expected), 196, 768) or
                    row["text"].shape != (768,) or
                    not torch.isfinite(row["patches"]).all() or
                    not torch.isfinite(row["text"]).all()):
                raise ValueError(f"visual record mismatch {part}/{record_id}")
            group_id = f"s{plan['seed']}_e{plan['episode_id']}"
            if group_id in text_by_group and \
                    not torch.equal(text_by_group[group_id], row["text"]):
                raise ValueError(f"instruction embedding mismatch {group_id}")
            text_by_group[group_id] = row["text"]
            scenes.add(plan["scene_id"])
            frames += len(expected)
        if (scenes & all_scenes or
                len(scenes) != manifest["inventory"][part]["scenes"]):
            raise ValueError("scene leakage or missing visual scene")
        all_scenes.update(scenes)
        report["parts"][part] = {"trajectories": len(plans),
                                 "frames": frames,
                                 "groups": len(text_by_group),
                                 "scenes": len(scenes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
