"""Independently verify adapted-encoder development cache provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from history_grounding_lora import digest
from fit_group4_future_success_linear import ANCHORS


def finite(value: torch.Tensor, shape: tuple[int, ...]) -> bool:
    return value.shape == shape and bool(torch.isfinite(value).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-record-root", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    source_id = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            checkpoint["schema"] != "group4_prefix_contrast_lora_v1" or \
            checkpoint["source_sha256"]["group_manifest"] != group_sha or \
            checkpoint["source_sha256"]["expert_manifest"] != expert_sha:
        raise ValueError("adapted encoder source mismatch")
    policy_states = 0
    policy_scenes = set()
    rows = group["selected"]["development"]
    for plan in rows:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        record_path = args.group_turn_root / "development" / "records" / f"{rid}.json"
        cache_path = args.group_state_root / "development" / "records" / f"{rid}.pt"
        record = json.loads(record_path.read_text())
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        last = max(t["original_turn_index"] for t in record["turns"])
        indices = [i for i, t in enumerate(record["turns"], 1)
                   if t["original_turn_index"] in ANCHORS and
                   t["original_turn_index"] < last]
        anchors = [record["turns"][i - 1]["original_turn_index"]
                   for i in indices]
        if record["record_id"] != rid or \
                record["manifest_sha256"] != group_sha or \
                record["scene_id"] != plan["scene_id"] or \
                record["terminal_mode"] != plan["terminal_mode"] or \
                cache["schema"] != "group_relative_state_cache_v1" or \
                cache["record_id"] != rid or \
                cache["source_id"] != source_id or \
                cache["manifest_sha256"] != group_sha or \
                cache["record_sha256"] != digest(record_path) or \
                cache["indices"] != indices or \
                cache["anchor_turns"] != anchors or \
                not finite(cache["hidden"], (len(anchors), 2048)):
            raise ValueError(f"bad policy cache {rid}")
        policy_states += len(anchors)
        policy_scenes.add(plan["scene_id"])
    expert_states = expert_contrasts = 0
    expert_scenes = set()
    for row in expert["selected"]["development"]:
        eid = str(row["episode_id"])
        record_path = args.expert_record_root / row["source_part"] / "records" / f"{eid}.json"
        cache_path = args.expert_state_root / "development" / "records" / f"{eid}.pt"
        record = json.loads(record_path.read_text())
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        anchors = [a for a in ANCHORS if a < row["turn_count"]]
        if digest(record_path) != row["record_sha256"] or \
                record["episode_id"] != eid or \
                record["scene_id"] != row["scene_id"] or \
                record["trajectory_id"] != row["trajectory_id"] or \
                record["turn_count"] != row["turn_count"] or \
                cache["schema"] != "group4_expert_prefix_state_v1" or \
                cache["episode_id"] != eid or \
                cache["part"] != "development" or \
                cache["source_id"] != source_id or \
                cache["manifest_sha256"] != expert_sha or \
                cache["record_sha256"] != row["record_sha256"] or \
                cache["anchors"] != anchors or \
                not finite(cache["correct"], (len(anchors), 2048)) or \
                not finite(cache["wrong"], (len(anchors), 2048)):
            raise ValueError(f"bad expert cache {eid}")
        expert_contrasts += len(anchors)
        expert_states += 2 * len(anchors)
        expert_scenes.add(row["scene_id"])
    if len(rows) != 160 or len(expert["selected"]["development"]) != 161 or \
            expert_contrasts != 303 or policy_scenes != expert_scenes or \
            len(policy_scenes) != 8:
        raise ValueError("development cache coverage or scene mismatch")
    result = {"schema": "group4_prefix_contrast_dev_cache_audit_v1",
              "source_sha256": {"group_manifest": group_sha,
                                "expert_manifest": expert_sha,
                                "checkpoint": source_id},
              "policy": {"trajectories": len(rows),
                         "preterminal_states": policy_states,
                         "scenes": len(policy_scenes)},
              "expert": {"trajectories": len(expert["selected"]["development"]),
                         "prefix_contrasts": expert_contrasts,
                         "states": expert_states,
                         "scenes": len(expert_scenes)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
