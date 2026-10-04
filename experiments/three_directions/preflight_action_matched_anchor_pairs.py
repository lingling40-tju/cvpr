"""Count visual hard negatives before allocating another reward-model fit.

This reads only the established fit/development RGB replay and replay
audits. It never uses val-unseen episodes or the opened turn-3 audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


MANIFEST_SHA = "4a0a2403fc4345308545d36f17102664856e24129eab189a32c0478a5bcf7967"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_meters(record: dict, anchor: int) -> float:
    total = 0.0
    for turn in record["input"]["action_history_by_anchor"][str(anchor)]:
        for action in turn["executed_actions"]:
            if action.startswith("move forward "):
                match = re.fullmatch(r"move forward (\d+)cm", action)
                if match is None:
                    raise ValueError(f"bad forward action: {action}")
                total += int(match.group(1)) / 100
    return total


def part(manifest: dict, root: Path, split: str) -> dict:
    grouped = defaultdict(list)
    for plan in manifest["selected"][split]:
        rid = plan["record_id"]
        record = json.loads((root / split / "records" / f"{rid}.json").read_text())
        audit = json.loads((root / split / "audits" / f"{rid}.json").read_text())
        if record.get("record_id") != rid or audit.get("record_id") != rid or \
                record.get("manifest_sha256") != MANIFEST_SHA or \
                audit.get("manifest_sha256") != MANIFEST_SHA:
            raise ValueError(f"source identity changed: {rid}")
        turns = {int(t["turn"]): t for t in audit["turns"]}
        grouped[(plan["seed"], str(plan["episode_id"]))].append(
            (plan, record, turns))
    result = {}
    for anchor in (3, 6):
        eligible = []
        for (seed, eid), group in grouped.items():
            group.sort(key=lambda row: row[0]["variant"])
            for i, (left, left_record, left_turns) in enumerate(group):
                for right, right_record, right_turns in group[i + 1:]:
                    if anchor not in left_turns or anchor not in right_turns or \
                            str(anchor) not in left_record["input"]["images"] or \
                            str(anchor) not in right_record["input"]["images"]:
                        continue
                    distance_gap = abs(float(left_turns[anchor]["after_distance_m"]) -
                                       float(right_turns[anchor]["after_distance_m"]))
                    if distance_gap < 1.0:
                        continue
                    action_gap = abs(forward_meters(left_record, anchor) -
                                     forward_meters(right_record, anchor))
                    eligible.append({"seed": seed, "episode": eid,
                                     "scene": left["scene_id"],
                                     "same_mode": left["terminal_mode"] ==
                                     right["terminal_mode"],
                                     "action_gap": action_gap,
                                     "distance_gap": distance_gap})
        def count(rows: list[dict]) -> dict:
            return {"pairs": len(rows),
                    "seed_episode_groups": len({(r["seed"], r["episode"])
                                                for r in rows}),
                    "unique_episode_ids": len({r["episode"] for r in rows}),
                    "scenes": len({r["scene"] for r in rows}),
                    "same_terminal_mode_pairs": sum(r["same_mode"]
                                                    for r in rows)}
        result[str(anchor)] = {
            "all_distance_gap_ge_1m": count(eligible),
            "forward_gap_le_0m": count([r for r in eligible
                                        if r["action_gap"] <= .0001]),
            "forward_gap_le_0p25m": count([r for r in eligible
                                           if r["action_gap"] <= .25]),
            "forward_gap_le_0p5m": count([r for r in eligible
                                          if r["action_gap"] <= .5]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != MANIFEST_SHA:
        raise ValueError("source manifest changed")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            manifest.get("group_size") != 4 or \
            manifest.get("seeds") != [11, 22]:
        raise ValueError("not the frozen group-four source")
    result = {
        "schema": "action_matched_anchor_pair_coverage_v1",
        "manifest_sha256": MANIFEST_SHA,
        "parts": {split: part(manifest, args.rgb_root, split)
                  for split in ("fit", "development")},
        "interpretation": "Coverage only; no reward-model accuracy or navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
