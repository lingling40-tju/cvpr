"""Exact-coverage and geodesic audit for frozen turn-3 RGB records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image

MANIFEST_SHA = "dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_meters(record: dict, anchor: int) -> float:
    meters = 0.0
    for turn in record["input"]["action_history_by_anchor"][str(anchor)]:
        for action in turn["executed_actions"]:
            if action.startswith("move forward "):
                match = re.fullmatch(r"move forward (\d+)cm", action)
                if match is None:
                    raise ValueError(f"unrecognized forward action: {action}")
                meters += int(match.group(1)) / 100.0
    return meters


def verify(manifest_path: Path, rgb_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != MANIFEST_SHA or \
            manifest.get("schema") != "early_anchor_group4_audit_source_manifest_v1" or \
            manifest.get("group_size") != 4 or \
            manifest.get("selected_records") != 540:
        raise ValueError("changed frozen audit manifest")
    root = rgb_root / "audit"
    plans = manifest["plans"]
    ids = {plan["record_id"] for plan in plans}
    if len(ids) != len(plans) or \
            {path.stem for path in (root / "records").glob("*.json")} != ids or \
            {path.stem for path in (root / "audits").glob("*.json")} != ids:
        raise ValueError("incomplete or extra audit records")
    summaries = [json.loads((root / f"summary.shard{shard}.json").read_text())
                 for shard in range(4)]
    if any(summary.get("schema") != "early_anchor_group4_audit_collection_v1" or
           summary.get("manifest_sha256") != MANIFEST_SHA or
           summary.get("shards") != 4 or summary.get("shard") != index or
           summary.get("errors") or
           summary.get("completed") != summary.get("requested")
           for index, summary in enumerate(summaries)) or \
            sum(summary["completed"] for summary in summaries) != len(plans):
        raise ValueError("audit shard summaries incomplete")
    by_group = defaultdict(list)
    scenes = set()
    max_drift = 0.0
    frames = 0
    for plan in plans:
        rid = plan["record_id"]
        record = json.loads((root / "records" / f"{rid}.json").read_text())
        audit = json.loads((root / "audits" / f"{rid}.json").read_text())
        if record.get("schema") != "future_advantage_sparse_model_input_v1" or \
                record.get("manifest_sha256") != MANIFEST_SHA or \
                record.get("record_id") != rid or \
                set(record) != {"schema", "manifest_sha256", "record_id", "input"} or \
                set(record["input"]) != {"instruction", "images",
                                         "action_history_by_anchor"} or \
                record["input"]["instruction"] != plan["instruction"] or \
                set(record["input"]["images"]) != {"0", "3"} or \
                set(record["input"]["action_history_by_anchor"]) != {"3"} or \
                len(record["input"]["action_history_by_anchor"]["3"]) != 3:
            raise ValueError(f"invalid observation-only audit input: {rid}")
        for relative in record["input"]["images"].values():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("audit image escaped root")
            with Image.open(root / path) as image:
                image.verify()
            frames += 1
        if audit.get("schema") != "future_advantage_sparse_replay_audit_v1" or \
                audit.get("manifest_sha256") != MANIFEST_SHA or \
                audit.get("record_id") != rid or \
                audit.get("seed") != plan["seed"] or \
                str(audit.get("episode_id")) != str(plan["episode_id"]) or \
                audit.get("variant") != plan["variant"] or \
                audit.get("scene_id") != plan["scene_id"]:
            raise ValueError(f"audit identity mismatch: {rid}")
        terminal = float(audit["terminal_distance_m"])
        source_terminal = float(plan["terminal_distance_m_for_replay_audit_only"])
        if not math.isfinite(terminal) or \
                abs(terminal - source_terminal) > .25:
            raise ValueError(f"audit terminal drift: {rid}")
        max_drift = max(max_drift, abs(terminal - source_terminal))
        turns = {int(turn["turn"]): turn for turn in audit["turns"]}
        if 3 not in turns or not math.isfinite(float(turns[3]["after_distance_m"])):
            raise ValueError(f"missing turn-3 distance audit: {rid}")
        by_group[(plan["seed"], str(plan["episode_id"]))].append(
            (plan, record, float(turns[3]["after_distance_m"])))
        scenes.add(plan["scene_id"])
    if frames != 2 * len(plans):
        raise ValueError("wrong audit frame count")
    pairs = []
    for (seed, eid), group in by_group.items():
        group.sort(key=lambda row: row[0]["variant"])
        for index, (left_plan, left_record, left_distance) in enumerate(group):
            for right_plan, right_record, right_distance in group[index + 1:]:
                gap = right_distance - left_distance
                if abs(gap) < 1.0:
                    continue
                sign = 1 if gap > 0 else -1
                action_gap = forward_meters(left_record, 3) - \
                    forward_meters(right_record, 3)
                pairs.append({
                    "seed": seed, "episode": eid,
                    "scene": left_plan["scene_id"],
                    "same_mode": left_plan["terminal_mode"] ==
                    right_plan["terminal_mode"],
                    "action_baseline": 1.0 if sign * action_gap > 0 else
                    (.5 if sign * action_gap == 0 else 0.0),
                })
    eligible_scenes = {pair["scene"] for pair in pairs}
    eligible_ids = {pair["episode"] for pair in pairs}
    return {
        "schema": "early_anchor_group4_audit_replay_verification_v1",
        "manifest_sha256": MANIFEST_SHA,
        "selected_records": len(plans),
        "rendered_frames": frames,
        "audit_scenes": len(scenes),
        "max_terminal_drift_m": max_drift,
        "eligible_distance_gap_ge_1m_pairs": len(pairs),
        "eligible_unique_episode_ids": len(eligible_ids),
        "eligible_scenes": len(eligible_scenes),
        "same_terminal_mode_pairs": sum(pair["same_mode"] for pair in pairs),
        "predeclared_minimum_coverage_met": bool(
            len(pairs) >= 100 and len(eligible_ids) >= 40 and
            len(eligible_scenes) >= 6),
        "interpretation": "Audit replay and comparison coverage only; no model ranking or navigation result",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest, args.rgb_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
