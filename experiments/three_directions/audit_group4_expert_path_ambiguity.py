"""Audit whether same-start wrong-goal expert paths share early RGB history.

This is a label-only provenance audit. It reads source images and records,
never model predictions, locked model-audit states, or val-unseen examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_path(root: Path, part: str, eid: str) -> Path:
    return root / part / "records" / f"{eid}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("incorrect expert manifest")
    cache: dict[str, str] = {}

    def frame_hash(root: Path, relative: str) -> str:
        path = root / relative
        key = str(path)
        if key not in cache:
            cache[key] = sha(path)
        return cache[key]

    summaries = {}
    for partition in ("fit", "development"):
        rows = manifest["selected"][partition]
        source_counts: Counter[str] = Counter()
        counts = {3: Counter(), 6: Counter()}
        eligible_counts = Counter()
        common_lengths = []
        missing_wrong_records = 0
        row_diagnostics = []
        for row in rows:
            eid, wrong_eid = str(row["episode_id"]), str(row["wrong_episode_id"])
            source_part = row["source_part"]
            original_path = record_path(args.record_root, source_part, eid)
            wrong_path = record_path(args.record_root, source_part, wrong_eid)
            if sha(original_path) != row["record_sha256"]:
                raise ValueError(f"source record hash mismatch: {eid}")
            source_counts[source_part] += 1
            if not wrong_path.exists():
                missing_wrong_records += 1
                row_diagnostics.append({
                    "episode_id": eid, "scene_id": row["scene_id"],
                    "wrong_episode_id": wrong_eid,
                    "anchor_status": {str(a): "wrong_record_unavailable"
                                      for a in (3, 6) if a < row["turn_count"]},
                })
                continue
            original = json.loads(original_path.read_text())
            wrong = json.loads(wrong_path.read_text())
            if original["episode_id"] != eid or wrong["episode_id"] != wrong_eid or \
                    original["scene_id"] != row["scene_id"] or \
                    wrong["scene_id"] != row["scene_id"] or \
                    original["wrong_instruction"].strip() != wrong["instruction"].strip():
                raise ValueError(f"wrong-goal pairing mismatch: {eid}")
            frame_root = args.record_root / source_part
            if frame_hash(frame_root, original["initial_image"]) != \
                    frame_hash(frame_root, wrong["initial_image"]):
                raise ValueError(f"initial RGB differs: {eid}")
            common = 0
            for a, b in zip(original["turns"], wrong["turns"]):
                if frame_hash(frame_root, a["image"]) != \
                        frame_hash(frame_root, b["image"]):
                    break
                common += 1
            common_lengths.append(common)
            anchor_status = {}
            for anchor in (3, 6):
                if anchor >= row["turn_count"]:
                    continue
                eligible_counts[anchor] += 1
                if len(wrong["turns"]) < anchor:
                    status = "wrong_route_too_short"
                elif common >= anchor:
                    status = "identical_rgb_history"
                else:
                    status = "visually_diverged"
                counts[anchor][status] += 1
                anchor_status[str(anchor)] = status
            row_diagnostics.append({
                "episode_id": eid, "scene_id": row["scene_id"],
                "wrong_episode_id": wrong_eid,
                "common_prefix_turns": common,
                "anchor_status": anchor_status,
            })
        for anchor in (3, 6):
            if sum(counts[anchor].values()) != eligible_counts[anchor]:
                raise ValueError(f"anchor-{anchor} coverage mismatch: {partition}")
        summaries[partition] = {
            "episodes": len(rows), "source_parts": dict(source_counts),
            "pairs_with_wrong_record": len(common_lengths),
            "pairs_missing_wrong_record": missing_wrong_records,
            "by_anchor": {str(a): dict(counts[a]) for a in (3, 6)},
            "common_prefix_turns": {
                "median": sorted(common_lengths)[len(common_lengths) // 2]
                if common_lengths else None,
                "at_least_3": sum(x >= 3 for x in common_lengths),
                "at_least_6": sum(x >= 6 for x in common_lengths),
            },
            "rows": row_diagnostics,
        }
    output = {
        "schema": "group4_expert_path_ambiguity_audit_v1",
        "interpretation": "exact JPEG history equality is a lower-bound ambiguity signal; no model result",
        "manifest_sha256": sha(args.manifest),
        "partitions": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({**output, "partitions": {
        part: {key: value for key, value in summary.items() if key != "rows"}
        for part, summary in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
