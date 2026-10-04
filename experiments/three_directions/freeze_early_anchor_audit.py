"""Freeze n=4 turn-3 audit routes without inspecting progress or scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


SPLIT_SHA = "c82e495e07f8eb0ed01ad0373662ac4cae82e10aaa3eedd9d467844844f4ba32"
SOURCE_MANIFEST_SHA = "4a0a2403fc4345308545d36f17102664856e24129eab189a32c0478a5bcf7967"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(info: dict) -> bool:
    turns = info["gen_traj"]
    return len(turns) >= 3 and all(
        turn.get("executed_actions") and all(
            str(action).strip().lower() != "stop"
            for action in turn["executed_actions"])
        for turn in turns[:3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if digest(args.scene_split) != SPLIT_SHA or \
            digest(args.source_manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError("frozen scene split or source manifest changed")
    source = json.loads(args.source_manifest.read_text())
    if source.get("group_size") != 4 or source.get("seeds") != [11, 22] or \
            digest(args.dataset) != source["source_sha256"]["dataset"]:
        raise ValueError("changed group-four train source")
    split = json.loads(args.scene_split.read_text())["scene_split"]
    audit_scenes = set(split["audit"])
    if len(audit_scenes) != 8 or audit_scenes & set(split["fit"]) or \
            audit_scenes & set(split["development"]):
        raise ValueError("audit scenes are not the frozen eight")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(stream)["episodes"]}
    plans = []
    source_hashes = {}
    group_counts = {}
    for seed in (11, 22):
        rollout = args.root / (
            f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl")
        train_audit = args.root / (
            f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json")
        hashes = {"rollout": digest(rollout), "train_audit": digest(train_audit)}
        if hashes != source["source_sha256"]["seeds"][str(seed)]:
            raise ValueError(f"source hash changed: seed {seed}")
        source_hashes[str(seed)] = hashes
        groups = defaultdict(list)
        with rollout.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    eid = str(info["episode_id"])
                    episode = episodes.get(eid)
                    if episode and episode["scene_id"] in audit_scenes:
                        groups[eid].append(info)
        kept_groups = 0
        for eid, rows in sorted(groups.items(), key=lambda item: int(item[0])):
            if len(rows) != 4:
                raise ValueError(f"audit episode is not n=4: {seed}/{eid}")
            choices = [(variant, info) for variant, info in enumerate(rows)
                       if eligible(info)]
            if len(choices) < 2:
                continue
            kept_groups += 1
            episode = episodes[eid]
            scene = episode["scene_id"]
            if scene.startswith("data/scene_datasets/"):
                scene = scene[len("data/scene_datasets/"):]
            for variant, info in choices:
                plans.append({
                    "record_id": f"s{seed}_e{eid}_v{variant}",
                    "seed": seed, "episode_id": eid, "variant": variant,
                    "scene_id": scene,
                    "instruction": info["instruction"].strip(),
                    "terminal_mode": info["end_reason"],
                    "terminal_distance_m_for_replay_audit_only": float(
                        info["distance_to_goal"]),
                    "anchor_turns": [3],
                })
        group_counts[str(seed)] = kept_groups
    plans.sort(key=lambda plan: (plan["scene_id"],
                                 int(plan["episode_id"]),
                                 plan["seed"], plan["variant"]))
    if len({plan["record_id"] for plan in plans}) != len(plans):
        raise ValueError("duplicate audit record")
    report = {
        "schema": "early_anchor_group4_audit_source_manifest_v1",
        "selection": "all seed-11/22 frozen-audit-scene n=4 routes with three executed movement turns; keep episode-seed group only if at least two routes qualify; no progress or model-score filtering",
        "group_size": 4, "seeds": [11, 22], "anchor_turns": [3],
        "source_sha256": {
            "scene_split": digest(args.scene_split),
            "source_manifest": digest(args.source_manifest),
            "dataset": digest(args.dataset), "seeds": source_hashes},
        "audit_scenes": len(audit_scenes),
        "groups_by_seed": group_counts,
        "selected_records": len(plans),
        "unique_episode_ids": len({plan["episode_id"] for plan in plans}),
        "plans": plans,
        "interpretation": "ID/action-length freeze only; turn-3 distance order, RGB, model scores, and grounding remain untested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.summary:
        summary = {
            "schema": "early_anchor_group4_audit_source_summary_v1",
            "manifest_sha256": digest(args.output),
            "source_sha256": report["source_sha256"],
            "group_size": 4, "anchor": 3,
            "audit_scenes": report["audit_scenes"],
            "groups_by_seed": report["groups_by_seed"],
            "selected_records": report["selected_records"],
            "unique_episode_ids": report["unique_episode_ids"],
            "interpretation": "Selection metadata only; no geodesic pair labels, RGB, or model scores",
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "audit_scenes", "groups_by_seed", "selected_records",
        "unique_episode_ids")}, indent=2))


if __name__ == "__main__":
    main()
