"""Freeze complete four-rollout groups for within-instruction progress ranks.

Selection uses scene and hashed group identity only. Intermediate
geodesic distances are replayed after this manifest is frozen; neither
terminal outcome nor terminal distance participates in selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from prepare_policy_process_manifest import digest, valid_info


TARGET_GROUPS = {"fit": 160, "development": 40, "audit": 40}
SALT = "group-four-relative-progress-v1|"


def order_key(group: dict) -> str:
    return hashlib.sha256(
        f"{SALT}{group['seed']}:{group['episode_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout) != 3:
        raise ValueError("expected the three complete group-four rollouts")
    split = json.loads(args.scene_split.read_text())
    if split["schema"] != "stop_history_lora_scene_split_v1":
        raise ValueError("wrong frozen scene split")
    scene_to_part = {scene: name for name, scenes in split["scene_split"].items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split["scene_split"].values())):
        raise ValueError("overlapping scene split")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row for row in json.load(stream)["episodes"]}
    eligible = defaultdict(lambda: defaultdict(list))
    sources = {}
    for seed, path in zip((11, 22, 33), args.rollout):
        steps, groups = [], defaultdict(list)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    eid = str(info["episode_id"])
                    if eid not in episodes:
                        raise ValueError(f"unknown train episode {eid}")
                    groups[eid].append(info)
        if steps != list(range(1, 129)) or len(groups) != 512 or \
                any(len(infos) != 4 for infos in groups.values()):
            raise ValueError(f"incomplete group-four source seed {seed}")
        sources[str(seed)] = {"path": str(path), "sha256": digest(path),
                              "episode_groups": 512, "rollouts": 2048}
        for eid, infos in groups.items():
            episode = episodes[eid]
            scene = str(episode["scene_id"])
            if scene not in scene_to_part or \
                    any(not valid_info(info) or
                        info["instruction"].strip() !=
                        episode["instruction"]["instruction_text"].strip()
                        for info in infos):
                continue
            plans = [{"seed": seed, "episode_id": eid,
                      "variant": variant, "scene_id": scene,
                      "trajectory_id": str(episode["trajectory_id"]),
                      "terminal_mode": str(info["end_reason"]),
                      "terminal_distance_m_for_replay_audit_only":
                          float(info["distance_to_goal"]),
                      "turns": sum(bool(turn.get("executed_actions"))
                                   for turn in info["gen_traj"])}
                     for variant, info in enumerate(infos)]
            group = {"group_id": f"s{seed}_e{eid}", "seed": seed,
                     "episode_id": eid, "scene_id": scene,
                     "plans": plans}
            eligible[scene_to_part[scene]][scene].append(group)
    selected_groups, selected_plans, inventory = {}, {}, {}
    for part, target in TARGET_GROUPS.items():
        scene_order = sorted(eligible[part])
        pools = {scene: sorted(eligible[part][scene], key=order_key)
                 for scene in scene_order}
        inventory[part] = {"scenes": len(scene_order),
                           "eligible_groups": sum(map(len, pools.values())),
                           "eligible_rollouts": 4 * sum(map(len, pools.values()))}
        chosen = []
        for position in range(max(map(len, pools.values()))):
            for scene in scene_order:
                if position < len(pools[scene]) and len(chosen) < target:
                    chosen.append(pools[scene][position])
            if len(chosen) >= target:
                break
        if len(chosen) != target:
            raise ValueError(f"underpowered {part} groups: {len(chosen)}/{target}")
        selected_groups[part] = [{key: value for key, value in group.items()
                                  if key != "plans"} for group in chosen]
        selected_plans[part] = [plan for group in chosen for plan in group["plans"]]
    group_ids = {group["group_id"] for groups in selected_groups.values()
                 for group in groups}
    if len(group_ids) != sum(TARGET_GROUPS.values()):
        raise ValueError("reused group across scene splits")
    if any(len(rows) != TARGET_GROUPS[part] * 4
           for part, rows in selected_plans.items()):
        raise ValueError("incomplete selected group of four")
    result = {"schema": "policy_group_relative_manifest_v1",
              "selection": "hash-ordered scene round robin over complete group-four rollouts, before intermediate distance replay",
              "scene_split_sha256": digest(args.scene_split),
              "train_dataset_sha256": digest(args.train_dataset),
              "sources": sources, "targets": TARGET_GROUPS,
              "inventory": inventory,
              "groups": selected_groups,
              "selected": selected_plans}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "selected_groups": {part: len(rows) for part, rows
                                          in selected_groups.items()},
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
