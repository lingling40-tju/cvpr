"""Inventory train-only action-turn progress labels for a learned reward.

The privileged geodesic deltas are supervision candidates only. This
script does not train a model, read val-unseen results, or treat repeated
rollouts of one episode as independent examples.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


EXPECTED_DATASET = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
EXPECTED_SPLIT = "c82e495e07f8eb0ed01ad0373662ac4cae82e10aaa3eedd9d467844844f4ba32"
THRESHOLDS = (0.25, 0.5, 1.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dataset) != EXPECTED_DATASET or \
            digest(args.scene_split) != EXPECTED_SPLIT:
        raise ValueError("R2R-train dataset or frozen scene split changed")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(stream)["episodes"]}
    split = json.loads(args.scene_split.read_text())
    scene_to_part = {scene: part
                     for part, scenes in split["scene_split"].items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split["scene_split"].values())):
        raise ValueError("overlapping scene partitions")
    counts = {part: Counter() for part in split["scene_split"]}
    episode_ids = {part: set() for part in counts}
    scene_ids = {part: set() for part in counts}
    both_sign_episode = {part: set() for part in counts}
    step_ids = []
    groups = 0
    for line in args.rollout.open():
        row = json.loads(line)
        step_ids.append(int(row["step"]))
        group = defaultdict(list)
        for info in row["info"]:
            eid = str(info["episode_id"])
            if eid not in episodes:
                raise ValueError(f"non-train episode {eid}")
            group[eid].append(info)
        if len(group) != 4 or any(len(values) != 4 for values in group.values()):
            raise ValueError("expected four n=4 episode groups per step")
        groups += len(group)
        for eid, infos in group.items():
            scene = str(episodes[eid]["scene_id"])
            part = scene_to_part.get(scene)
            if part is None:
                raise ValueError(f"scene outside frozen split: {scene}")
            episode_ids[part].add(eid)
            scene_ids[part].add(scene)
            group_positive = group_negative = False
            for info in infos:
                counts[part]["rollouts"] += 1
                turns = info["gen_traj"]
                for turn in turns:
                    counts[part]["generated_turns"] += 1
                    progress = float(turn["oracle_turn_progress"])
                    if not math.isfinite(progress):
                        raise ValueError("nonfinite oracle turn progress")
                    executed = bool(turn["executed_actions"])
                    stop = bool(turn["oracle_stop_response"])
                    if not executed:
                        counts[part]["unexecuted_turns"] += 1
                        if abs(progress) > 1e-6:
                            raise ValueError("unexecuted turn has progress")
                        continue
                    counts[part]["executed_turns"] += 1
                    if stop:
                        counts[part]["generated_stop_turns"] += 1
                        if abs(progress) > 1e-6:
                            raise ValueError("generated STOP has progress")
                        continue
                    counts[part]["action_turns_for_representation"] += 1
                    before = float(turn["oracle_before_distance"])
                    after = float(turn["oracle_after_distance"])
                    if not math.isfinite(before) or not math.isfinite(after):
                        raise ValueError("nonfinite geodesic distance")
                    delta_m = before - after
                    if (abs(delta_m) > 1e-6) != (abs(progress) > 1e-6) or \
                            delta_m * progress < 0:
                        raise ValueError("reward and geodesic progress disagree")
                    if abs(progress) > 1e-6:
                        counts[part]["nonzero_action_turns"] += 1
                    if abs(delta_m) > 1e-6:
                        counts[part]["nonzero_geodesic_action_turns"] += 1
                    for threshold in THRESHOLDS:
                        label = str(threshold)
                        if delta_m >= threshold:
                            counts[part][f"positive_ge_{label}m"] += 1
                            group_positive = True
                        elif delta_m <= -threshold:
                            counts[part][f"negative_le_minus_{label}m"] += 1
                            group_negative = True
            if group_positive and group_negative:
                both_sign_episode[part].add(eid)
    if step_ids != list(range(1, 65)) or groups != 256 or \
            sum(len(values) for values in episode_ids.values()) != 256:
        raise ValueError("incomplete or repeated 64-step oracle groups")
    report = {
        "schema": "oracle_turn_label_preflight_v2",
        "source_sha256": {"rollout": digest(args.rollout),
                          "dataset": EXPECTED_DATASET,
                          "scene_split": EXPECTED_SPLIT},
        "steps": len(step_ids), "groups": groups,
        "parts": {part: {"unique_episodes": len(episode_ids[part]),
                         "scenes": len(scene_ids[part]),
                         "episodes_with_both_progress_signs":
                             len(both_sign_episode[part]),
                         "counts": dict(sorted(counts[part].items()))}
                  for part in counts},
        "interpretation": (
            "Train-only privileged geodesic labels in raw meters; counts show data availability, "
            "not learned-reward quality or a navigation gain. Cluster by scene and "
            "episode before comparing models."
        )}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
