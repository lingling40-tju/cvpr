"""Select fixed R2R-train episodes for RL-stage development and reservation.

This only freezes episode identities. The reserved screen must not be used to
select a method or tune a threshold, and neither screen is a clean test of
scene generalization from the common SFT initialization.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path


SALT = "trajectory-sil-rl-screen-20261006-v1/"
COUNT = 256
MIN_PER_SCENE = 4


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select(role: str, scenes: list[str], by_scene: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    selected = []
    for scene in scenes:
        ranked = sorted(by_scene[scene], key=lambda episode_id: digest(SALT + role + "/" + episode_id))
        if len(ranked) < MIN_PER_SCENE:
            raise ValueError("too few episodes in scene " + scene)
        selected.extend(ranked[:MIN_PER_SCENE])
    selected_set = set(selected)
    remaining = [episode_id for scene in scenes for episode_id in by_scene[scene]
                 if episode_id not in selected_set]
    remaining.sort(key=lambda episode_id: digest(SALT + role + "/" + episode_id))
    selected.extend(remaining[:COUNT - len(selected)])
    selected.sort(key=lambda episode_id: digest(SALT + role + "/order/" + episode_id))
    if len(selected) != COUNT or len(set(selected)) != COUNT:
        raise ValueError("screen episode coverage mismatch")
    return selected, scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.episodes.read_bytes()
    split_bytes = args.scene_split.read_bytes()
    split = json.loads(split_bytes)
    if hashlib.sha256(source_bytes).hexdigest() != split["source_sha256"]:
        raise ValueError("R2R-train source hash differs from frozen scene split")
    records = json.loads(gzip.decompress(source_bytes))["episodes"]
    by_scene = defaultdict(list)
    scene_of = {}
    for item in records:
        episode_id, scene = str(item["episode_id"]), str(item["scene_id"])
        if episode_id in scene_of:
            raise ValueError("duplicate R2R-train episode ID")
        scene_of[episode_id] = scene
        by_scene[scene].append(episode_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_selected = set()
    for role in ("development", "reserved"):
        scenes = split["partitions"][role]["scenes"]
        ids, _ = select(role, scenes, by_scene)
        if all_selected.intersection(ids):
            raise ValueError("development and reserved episodes overlap")
        all_selected.update(ids)
        observed = [scene_of[episode_id] for episode_id in ids]
        if set(observed) != set(scenes) or min(observed.count(scene) for scene in scenes) < MIN_PER_SCENE:
            raise ValueError("screen scene coverage mismatch")
        manifest = {
            "schema": "trajectory_sil_rl_train_screen_v1",
            "split": "train",
            "role": role,
            "source_sha256": split["source_sha256"],
            "scene_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "selection_salt": SALT,
            "episode_ids": ids,
            "scene_ids": observed,
            "limitation": "R2R-train RL-stage screen; common SFT initialization and prior exploratory work may have seen these scenes.",
        }
        output = args.output_dir / (role + "256.json")
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(role, len(ids), len(set(observed)), hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
