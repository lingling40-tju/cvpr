"""Verify exact expert prefix cache coverage and source provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from history_grounding_lora import digest
from fit_group4_future_success_linear import ANCHORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    source_id = digest(args.checkpoint)
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("wrong joint expert manifest")
    all_scenes = set()
    report = {"schema": "group4_expert_prefix_state_audit_v1",
              "manifest_sha256": manifest_sha, "source_id": source_id,
              "parts": {}}
    for part, shards in (("fit", 3), ("development", 1)):
        rows = manifest["selected"][part]
        for shard in range(shards):
            path = args.cache_root / part / f"summary_shard{shard}of{shards}.json"
            summary = json.loads(path.read_text())
            expected = rows[shard::shards]
            if summary["schema"] != "group4_expert_prefix_state_shard_v1" or \
                    summary["part"] != part or summary["shard"] != shard or \
                    summary["shards"] != shards or \
                    summary["requested"] != len(expected) or \
                    summary["completed"] != len(expected) or \
                    summary["manifest_sha256"] != manifest_sha or \
                    summary["source_id"] != source_id:
                raise ValueError(f"incomplete prefix state shard {part}/{shard}")
        scenes = set()
        by_anchor = {str(anchor): 0 for anchor in ANCHORS}
        for row in rows:
            eid = str(row["episode_id"])
            path = args.cache_root / part / "records" / f"{eid}.pt"
            cache = torch.load(path, map_location="cpu", weights_only=True)
            anchors = [anchor for anchor in ANCHORS
                       if anchor < row["turn_count"]]
            if cache["schema"] != "group4_expert_prefix_state_v1" or \
                    cache["episode_id"] != eid or cache["part"] != part or \
                    cache["manifest_sha256"] != manifest_sha or \
                    cache["source_id"] != source_id or \
                    cache["record_sha256"] != row["record_sha256"] or \
                    cache["anchors"] != anchors or \
                    cache["correct"].shape != (len(anchors), 2048) or \
                    cache["wrong"].shape != (len(anchors), 2048) or \
                    not bool(torch.isfinite(cache["correct"]).all()) or \
                    not bool(torch.isfinite(cache["wrong"]).all()):
                raise ValueError(f"invalid expert prefix state {part}/{eid}")
            for anchor in anchors:
                by_anchor[str(anchor)] += 1
            scenes.add(row["scene_id"])
        if scenes & all_scenes or len(scenes) != manifest["inventory"][part]["scenes"]:
            raise ValueError(f"expert prefix scene leakage/missing scene {part}")
        all_scenes.update(scenes)
        report["parts"][part] = {"expert_episodes": len(rows),
                                 "instruction_prefix_contrasts":
                                     sum(by_anchor.values()),
                                 "states": 2 * sum(by_anchor.values()),
                                 "by_anchor": by_anchor,
                                 "scenes": len(scenes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
