"""Select train-only expert paths for an ordinal visual-progress pilot.

Selection is deterministic, scene-disjoint between fit and calibration, and
contains natural same-start/different-goal instruction pairs. This writes a
manifest of IDs and metadata only; frame collection is a separate GPU task.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_turn(actions: list[int]) -> int:
    yaw = 0
    for action in actions:
        if action in (0, 1):
            break
        if action not in (2, 3):
            raise ValueError(f"invalid action {action}")
        yaw += 1 if action == 3 else -1
    return yaw


def candidate_pairs(episodes: list[dict], ground_truth: dict) -> dict[str, list[tuple]]:
    starts = defaultdict(list)
    for episode in episodes:
        pose = (
            episode["scene_id"],
            tuple(round(float(v), 3) for v in episode["start_position"]),
            tuple(round(float(v), 3) for v in episode["start_rotation"]),
        )
        starts[pose].append(episode)
    by_scene = defaultdict(list)
    for (scene, _, _), rows in starts.items():
        for left, right in itertools.combinations(rows, 2):
            distance = math.dist(left["goals"][0]["position"], right["goals"][0]["position"])
            if distance <= 3:
                continue
            left_yaw = first_turn(ground_truth[str(left["episode_id"])]["actions"])
            right_yaw = first_turn(ground_truth[str(right["episode_id"])]["actions"])
            if left_yaw * right_yaw < 0 and min(abs(left_yaw), abs(right_yaw)) >= 2:
                a, b = sorted((int(left["episode_id"]), int(right["episode_id"])))
                by_scene[scene].append((-distance, a, b))
    for scene in by_scene:
        by_scene[scene].sort()
    return by_scene


def choose_pairs(scenes: list[str], candidates: dict[str, list[tuple]], count: int):
    selected = []
    used = set()
    offsets = {scene: 0 for scene in scenes}
    while len(selected) < count:
        added = False
        for scene in scenes:
            rows = candidates.get(scene, [])
            while offsets[scene] < len(rows):
                _, a, b = rows[offsets[scene]]
                offsets[scene] += 1
                if a not in used and b not in used:
                    selected.append((scene, a, b))
                    used.update((a, b))
                    added = True
                    break
            if len(selected) == count:
                break
        if not added:
            raise ValueError(f"only {len(selected)} disjoint pairs; requested {count}")
    return selected, used


def choose_filler(episodes: list[dict], scenes: list[str], used: set[int], count: int):
    by_scene = defaultdict(list)
    scene_set = set(scenes)
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        if episode["scene_id"] in scene_set and episode_id not in used:
            by_scene[episode["scene_id"]].append(episode_id)
    for scene in by_scene:
        by_scene[scene].sort()
    output = []
    round_index = 0
    while len(output) < count:
        added = False
        for scene in scenes:
            if round_index < len(by_scene[scene]):
                output.append(by_scene[scene][round_index])
                added = True
                if len(output) == count:
                    break
        if not added:
            raise ValueError(f"only {len(output)} filler episodes; requested {count}")
        round_index += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--val-unseen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = read_gzip_json(args.train)["episodes"]
    ground_truth = read_gzip_json(args.ground_truth)
    val_episodes = read_gzip_json(args.val_unseen)["episodes"]
    by_id = {int(item["episode_id"]): item for item in episodes}
    if len(by_id) != len(episodes) or set(map(str, by_id)) != set(ground_truth):
        raise ValueError("train episodes and ground truth disagree")
    scenes = sorted({row["scene_id"] for row in episodes},
                    key=lambda name: hashlib.sha256(("ordinal-v1:" + name).encode()).hexdigest())
    if len(scenes) != 61:
        raise ValueError(f"expected 61 train scenes, got {len(scenes)}")
    val_scenes = {item["scene_id"] for item in val_episodes}
    if set(scenes) & val_scenes:
        raise ValueError("train and val-unseen scenes overlap")
    candidates = candidate_pairs(episodes, ground_truth)
    # Spread the counterfactual calibration test over all held-out scenes.
    calibration_scenes = [scene for scene in scenes if candidates.get(scene)][:10]
    if len(calibration_scenes) != 10:
        raise ValueError("fewer than ten scenes have counterfactual pairs")
    calibration_set = set(calibration_scenes)
    fit_scenes = [scene for scene in scenes if scene not in calibration_set]
    output = {
        "version": 1,
        "interpretation": "Train-only representation fit/calibration manifest; no model score or navigation result.",
        "sources": {"train_sha256": sha256(args.train),
                    "ground_truth_sha256": sha256(args.ground_truth),
                    "val_unseen_sha256": sha256(args.val_unseen)},
        "fit_scenes": fit_scenes,
        "calibration_scenes": calibration_scenes,
        "scene_overlap": 0,
        "subsets": {},
    }
    all_selected = set()
    for label, subset_scenes, pair_count, target_count in (
        ("fit", fit_scenes, 128, 512),
        ("calibration", calibration_scenes, 32, 128),
    ):
        pairs, paired_ids = choose_pairs(subset_scenes, candidates, pair_count)
        filler = choose_filler(episodes, subset_scenes, paired_ids, target_count - 2 * pair_count)
        selected = paired_ids | set(filler)
        if len(selected) != target_count or selected & all_selected:
            raise ValueError("episode count or split disjointness failed")
        all_selected.update(selected)
        output["subsets"][label] = {
            "episode_ids": sorted(selected),
            "pairs": [{"scene": scene, "left": a, "right": b}
                      for scene, a, b in pairs],
            "episodes": target_count,
            "pairs_count": pair_count,
            "scenes_with_episodes": len({by_id[i]["scene_id"] for i in selected}),
            "scenes_with_pairs": len({scene for scene, _, _ in pairs}),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "fit": {k: v for k, v in output["subsets"]["fit"].items()
                              if k not in ("episode_ids", "pairs")},
                      "calibration": {k: v for k, v in output["subsets"]["calibration"].items()
                                      if k not in ("episode_ids", "pairs")},
                      "scene_overlap": output["scene_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
