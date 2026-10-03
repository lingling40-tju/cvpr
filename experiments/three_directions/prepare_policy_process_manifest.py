"""Freeze group-four policy trajectories for regression-sensitive progress.

Uses only completed R2R-train rollouts and a frozen scene split. It selects
scene-balanced trajectory identities before any intermediate distances are
replayed or scored. Multiple rollouts can share an episode; subsequent
statistics must cluster by scene/episode, never treat them as independent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


TARGET = {"fit": 768, "development": 320, "audit": 320}
SALT = "policy-process-regression-v1|"
VALID = {"stop"} | {f"move forward {v}cm" for v in (25, 50, 75)} | {
    f"turn {direction} {v} degrees" for direction in ("left", "right")
    for v in (15, 30, 45)}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(plan: dict) -> str:
    text = f"{SALT}{plan['seed']}:{plan['episode_id']}:{plan['variant']}"
    return hashlib.sha256(text.encode()).hexdigest()


def valid_info(info: dict) -> bool:
    if info.get("data_source") != "r2r" or info.get("action_space") != "r2r" or \
            info.get("global_start_step") != 1:
        return False
    turns = info.get("gen_traj", [])
    # The trainer records a 13th, unexecuted response when the 12-turn limit
    # is exceeded. Retain these timeouts and replay only their executed turns.
    timeout_bookkeeping = (len(turns) == 13 and
                           info.get("end_reason") == "number of turns exceeded." and
                           turns[-1].get("executed_actions") == [])
    if len(turns) < 3 or (len(turns) > 12 and not timeout_bookkeeping):
        return False
    actions = [str(action) for turn in turns
               for action in turn.get("executed_actions", [])]
    if not actions or any(action not in VALID for action in actions) or \
            "stop" in actions[:-1]:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout) != 3:
        raise ValueError("expected three group-four seed rollouts")
    split = json.loads(args.scene_split.read_text())
    if split["schema"] != "stop_history_lora_scene_split_v1":
        raise ValueError("wrong frozen scene split")
    scene_to_part = {scene: name for name, scenes in split["scene_split"].items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split["scene_split"].values())):
        raise ValueError("overlapping scenes")
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
                any(len(values) != 4 for values in groups.values()):
            raise ValueError(f"incomplete seed {seed} group-four rollout")
        sources[str(seed)] = {"path": str(path), "sha256": digest(path),
                              "episode_groups": len(groups), "rollouts": 2048}
        for eid, infos in groups.items():
            episode = episodes[eid]
            scene = str(episode["scene_id"])
            if scene not in scene_to_part:
                continue
            part = scene_to_part[scene]
            for variant, info in enumerate(infos):
                if not valid_info(info) or info["instruction"].strip() != \
                        episode["instruction"]["instruction_text"].strip():
                    continue
                plan = {"seed": seed, "episode_id": eid,
                        "variant": variant, "scene_id": scene,
                        "trajectory_id": str(episode["trajectory_id"]),
                        "terminal_mode": str(info["end_reason"]),
                        "terminal_distance_m_for_replay_audit_only":
                            float(info["distance_to_goal"]),
                        "turns": sum(bool(turn.get("executed_actions"))
                                     for turn in info["gen_traj"])}
                eligible[part][scene].append(plan)
    selected = {}
    inventory = {}
    for part, target in TARGET.items():
        scene_order = sorted(eligible[part])
        pools = {scene: sorted(eligible[part][scene], key=key)
                 for scene in scene_order}
        inventory[part] = {"scenes": len(scene_order),
                           "eligible": sum(map(len, pools.values())),
                           "terminal_modes": dict(Counter(row["terminal_mode"]
                                                          for pool in pools.values()
                                                          for row in pool))}
        chosen = []
        for position in range(max(map(len, pools.values()))):
            for scene in scene_order:
                if position < len(pools[scene]) and len(chosen) < target:
                    chosen.append(pools[scene][position])
            if len(chosen) >= target:
                break
        if len(chosen) != target:
            raise ValueError(f"underpowered {part}: {len(chosen)}/{target}")
        selected[part] = chosen
    identities = {(row["seed"], row["episode_id"], row["variant"])
                  for plans in selected.values() for row in plans}
    if len(identities) != sum(TARGET.values()):
        raise ValueError("reused policy trajectory across scene split")
    result = {
        "schema": "policy_process_train_manifest_v1",
        "selection": "hash-ordered scene round robin, before intermediate distance replay",
        "scene_split_sha256": digest(args.scene_split),
        "train_dataset_sha256": digest(args.train_dataset),
        "sources": sources, "targets": TARGET,
        "inventory": inventory,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "selected": {part: len(plans) for part, plans in selected.items()},
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
