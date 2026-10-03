"""Verify cached policy-turn states, source identity, and split coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from history_grounding_lora import digest, rid, signed_pairs


def indices(record: dict) -> list[int]:
    distances = [record["start_distance_to_goal_for_label_only"]] + [
        turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
    pairs = signed_pairs(distances)
    afters = set(pairs["backward"])
    afters.update(pairs["forward"][:2])
    return sorted({index for after in afters for index in (after - 1, after)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.policy_manifest.read_text())
    manifest_sha = digest(args.policy_manifest)
    result = {"schema": "pairwise_policy_state_cache_audit_v1",
              "policy_manifest_sha256": manifest_sha,
              "source_id": args.source_id, "parts": {}}
    for part, shards in (("fit", 2), ("development", 1)):
        records = []
        for plan in manifest["selected"][part]:
            row = json.loads((args.policy_root / part / "records" /
                              f"{rid(plan)}.json").read_text())
            if row["record_id"] != rid(plan) or row["manifest_sha256"] != manifest_sha:
                raise ValueError(f"source record mismatch {part}/{rid(plan)}")
            state_indices = indices(row)
            if state_indices:
                records.append((rid(plan), state_indices))
        records.sort()
        summary_counts = []
        states = 0
        for shard in range(shards):
            summary = json.loads((args.cache_root / part /
                                  f"summary_shard{shard}of{shards}.json").read_text())
            expected = records[shard::shards]
            if summary["policy_manifest_sha256"] != manifest_sha or \
                    summary["source_id"] != args.source_id or \
                    summary["part"] != part or \
                    summary["requested"] != len(expected) or \
                    summary["completed"] != len(expected) or \
                    summary["limit"] != 0:
                raise ValueError(f"shard summary mismatch {part}/{shard}")
            summary_counts.append(len(expected))
            for record_id, expected_indices in expected:
                row = torch.load(args.cache_root / part / "records" /
                                 f"{record_id}.pt", map_location="cpu",
                                 weights_only=True)
                if row["schema"] != "pairwise_policy_state_cache_v1" or \
                        row["record_id"] != record_id or \
                        row["source_id"] != args.source_id or \
                        row["policy_manifest_sha256"] != manifest_sha or \
                        row["indices"] != expected_indices or \
                        row["hidden"].shape != (len(expected_indices), 2048) or \
                        not torch.isfinite(row["hidden"]).all():
                    raise ValueError(f"feature cache mismatch {part}/{record_id}")
                states += len(expected_indices)
        result["parts"][part] = {"eligible_trajectories": len(records),
                                  "cached_states": states,
                                  "shard_trajectories": summary_counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
