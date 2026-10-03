"""Merge complete, disjoint replay shards before the turn-label audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from collect_policy_preference_frames import digest
from collect_policy_process_turns import group_shard, record_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    plans = manifest["selected"][args.part]
    folder = args.output_root / args.part
    reports = []
    for shard in range(args.shards):
        name = "summary.json" if args.shards == 1 else f"summary.shard{shard}.json"
        report = json.loads((folder / name).read_text())
        expected = sum(group_shard(plan, args.shards) == shard for plan in plans)
        if (report["part"] != args.part or report["manifest_sha256"] != manifest_sha
                or report["requested"] != expected or report["completed"] != expected
                or report["errors"] or report["smoke_limit"]
                or report["shard"] != shard or report["shards"] != args.shards):
            raise ValueError(f"incomplete or mismatched shard {args.part}/{shard}")
        reports.append(report)
    ids = set()
    for plan in plans:
        rid = record_id(plan)
        record = json.loads((folder / "records" / f"{rid}.json").read_text())
        identity = (record["seed"], record["episode_id"], record["variant"])
        if identity in ids or identity != (plan["seed"], str(plan["episode_id"]),
                                           plan["variant"]):
            raise ValueError(f"duplicate or wrong record {rid}")
        ids.add(identity)
    modes = Counter()
    for report in reports:
        modes.update(report["terminal_modes"])
    combined = {"schema": "policy_process_turn_collection_v1",
                "part": args.part, "manifest_sha256": manifest_sha,
                "requested": len(plans), "completed": len(ids),
                "unique_episodes": len({plan["episode_id"] for plan in plans}),
                "terminal_modes": dict(modes),
                "progress_turns_geodesic_1m": sum(
                    report["progress_turns_geodesic_1m"] for report in reports),
                "regression_turns_geodesic_1m": sum(
                    report["regression_turns_geodesic_1m"] for report in reports),
                "reused_from_verified_replay": sum(
                    report["reused_from_verified_replay"] for report in reports),
                "errors": [], "smoke_limit": 0, "shards": args.shards}
    (folder / "summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
