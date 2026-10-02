"""Select diverse natural train-only goal pairs for a larger visual screen.

The scene split is inherited from the frozen ordinal manifest. Selection uses
only train metadata and deterministic hashes, before reading any image score.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


DATASET = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def order_key(value: str) -> str:
    return hashlib.sha256(("goal-view-expanded-v1:" + value).encode()).hexdigest()


def pose_key(episode: dict) -> tuple:
    return (tuple(round(x, 4) for x in episode["start_position"]),
            tuple(round(x, 4) for x in episode["start_rotation"]))


def scene_selection(episodes: list[dict], cap: int) -> tuple[list[int], list[dict]]:
    scene = episodes[0]["scene_id"]
    by_pose = defaultdict(list)
    for episode in episodes:
        by_pose[pose_key(episode)].append(episode)
    candidates = []
    for pose, rows in by_pose.items():
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                if math.dist(left["goals"][0]["position"],
                             right["goals"][0]["position"]) >= 2.0:
                    a, b = sorted((int(left["episode_id"]), int(right["episode_id"])))
                    candidates.append((pose, a, b))
    candidates.sort(key=lambda row: order_key(f"{scene}:{row[1]}:{row[2]}"))
    selected, selected_by_pose = set(), defaultdict(set)
    # Cover many distinct physical starts before allocating extra goals to a
    # single start. A second pass fills the cap with additional hard pairs.
    for first_pass in (True, False):
        for pose, a, b in candidates:
            at_pose = selected_by_pose[pose]
            if first_pass and at_pose:
                continue
            added = {a, b} - selected
            if len(selected) + len(added) > cap or len(at_pose | {a, b}) > 8:
                continue
            selected.update((a, b))
            at_pose.update((a, b))
    pairs = [{"left": a, "right": b, "scene": scene}
             for _pose, a, b in candidates if a in selected and b in selected]
    return sorted(selected), pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-cap-per-scene", type=int, default=80)
    parser.add_argument("--calibration-cap-per-scene", type=int, default=40)
    args = parser.parse_args()
    original = json.loads(args.ordinal_manifest.read_text())
    source_hash = digest(DATASET)
    if source_hash != original["sources"]["train_sha256"]:
        raise ValueError("train source changed")
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        all_episodes = json.load(stream)["episodes"]
    by_scene = defaultdict(list)
    for episode in all_episodes:
        by_scene[episode["scene_id"]].append(episode)
    if set(by_scene) != set(original["fit_scenes"]) | set(original["calibration_scenes"]):
        raise ValueError("train scene universe mismatch")
    output = {"schema": "train_goal_views_v1", "sources": original["sources"],
              "original_manifest_sha256": digest(args.ordinal_manifest),
              "selection": {"goal_separation_m": 2.0, "max_goals_per_start": 8,
                            "fit_cap_per_scene": args.fit_cap_per_scene,
                            "calibration_cap_per_scene": args.calibration_cap_per_scene},
              "fit_scenes": original["fit_scenes"],
              "calibration_scenes": original["calibration_scenes"], "subsets": {}}
    for subset in ("fit", "calibration"):
        cap = (args.fit_cap_per_scene if subset == "fit"
               else args.calibration_cap_per_scene)
        ids, pairs, summary = [], [], []
        for scene in original[f"{subset}_scenes"]:
            chosen, related = scene_selection(by_scene[scene], cap)
            ids.extend(chosen)
            pairs.extend(related)
            summary.append({"scene": scene, "episodes": len(chosen),
                            "pairs": len(related)})
        ids.sort()
        pairs.sort(key=lambda row: (row["scene"], row["left"], row["right"]))
        if len(ids) != len(set(ids)) or not pairs:
            raise ValueError("selection coverage failure")
        if subset == "calibration" and any(row["pairs"] == 0 for row in summary):
            raise ValueError("calibration scene without natural goal pair")
        output["subsets"][subset] = {"episode_ids": ids, "pairs": pairs,
                                     "scenes": summary}
    if set(output["subsets"]["fit"]["episode_ids"]) & \
            set(output["subsets"]["calibration"]["episode_ids"]):
        raise ValueError("fit/calibration episode overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: {"episodes": len(output["subsets"][key]["episode_ids"]),
                            "pairs": len(output["subsets"][key]["pairs"]),
                            "scenes": sum(row["pairs"] > 0 for row in output["subsets"][key]["scenes"])}
                      for key in ("fit", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
