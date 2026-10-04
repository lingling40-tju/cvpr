"""Inventory train-scene supervision for a future group-relative critic.

At an anchor turn, compare the *future* oracle return of two rollouts of
the same episode. The candidate critic would see only the instruction,
RGB/action history through that anchor, and the other group histories.
Privileged geodesic returns are labels for this preflight, never inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


EXPECTED_ROLLOUT = "9b9371b221bdf2106cd3ac653de606c6286cd0dd67386e28936d3adb4488fe98"
EXPECTED_DATASET = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
EXPECTED_SPLIT = "c82e495e07f8eb0ed01ad0373662ac4cae82e10aaa3eedd9d467844844f4ba32"
FORWARD = re.compile(r"move forward ([0-9]+)cm")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def commanded_forward_meters(turns: list[dict], anchor: int) -> float:
    centimeters = 0
    for turn in turns[:anchor]:
        for action in turn["executed_actions"]:
            action = str(action).strip().lower()
            if action.startswith("move forward"):
                match = FORWARD.fullmatch(action)
                if match is None:
                    raise ValueError(f"unrecognized forward command: {action}")
                centimeters += int(match.group(1))
    return centimeters / 100.0


def point(left: float, right: float, future_gap: float) -> float:
    if left == right:
        return .5
    return float((left > right) == (future_gap > 0))


def summarize(rows: list[dict]) -> dict:
    episodes = defaultdict(list)
    scenes = defaultdict(dict)
    for row in rows:
        key = (row["scene"], row["episode_id"])
        episodes[key].append(row)
    for (scene, eid), pairs in episodes.items():
        scenes[scene][eid] = {
            "forward_prefix": statistics.mean(x["forward_prefix"] for x in pairs),
            "oracle_past_progress": statistics.mean(x["oracle_past_progress"]
                                                      for x in pairs),
        }
    return {
        "pairs": len(rows),
        "episode_groups": len(episodes),
        "scenes": len(scenes),
        "median_abs_future_return_gap": statistics.median(
            abs(row["future_gap"]) for row in rows) if rows else None,
        "forward_prefix_baseline": {
            "pair_accuracy_with_half_ties": statistics.mean(
                row["forward_prefix"] for row in rows) if rows else None,
            "episode_macro_accuracy": statistics.mean(
                statistics.mean(row["forward_prefix"] for row in pairs)
                for pairs in episodes.values()) if episodes else None,
            "scene_macro_accuracy": statistics.mean(
                statistics.mean(value["forward_prefix"]
                                for value in episode.values())
                for episode in scenes.values()) if scenes else None,
        },
        "privileged_past_progress_diagnostic": {
            "episode_macro_accuracy": statistics.mean(
                statistics.mean(row["oracle_past_progress"] for row in pairs)
                for pairs in episodes.values()) if episodes else None,
        },
    }


def inventory(rollout: Path, dataset: Path, scene_split: Path,
              *, expected_rollout_sha: str | None = EXPECTED_ROLLOUT,
              expected_dataset_sha: str = EXPECTED_DATASET,
              expected_steps: int = 64,
              expected_episodes: int = 256,
              include_same_mode: bool = False) -> dict:
    source = {"rollout": digest(rollout), "dataset": digest(dataset),
              "scene_split": digest(scene_split)}
    if (expected_rollout_sha is not None and
            source["rollout"] != expected_rollout_sha) or \
            source["dataset"] != expected_dataset_sha or \
            source["scene_split"] != EXPECTED_SPLIT:
        raise ValueError("frozen oracle rollout or R2R-train sources changed")
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(stream)["episodes"]}
    split = json.loads(scene_split.read_text())["scene_split"]
    scene_to_part = {scene: part for part, scenes in split.items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split.values())):
        raise ValueError("overlapping scene partitions")
    rows = {part: {3: [], 6: []} for part in split}
    same_mode_rows = {part: {3: [], 6: []} for part in split}
    all_failure = {part: set() for part in split}
    seen_episodes = set()
    steps = []
    for line in rollout.open():
        batch = json.loads(line)
        steps.append(batch["step"])
        groups = defaultdict(list)
        for info in batch["info"]:
            groups[str(info["episode_id"])].append(info)
        if len(groups) != 4 or any(len(group) != 4 for group in groups.values()):
            raise ValueError("expected four episodes with four rollouts each")
        for eid, infos in groups.items():
            if eid in seen_episodes:
                raise ValueError("episode repeated across steps")
            seen_episodes.add(eid)
            if eid not in episodes:
                raise ValueError(f"non-train episode {eid}")
            scene = str(episodes[eid]["scene_id"])
            part = scene_to_part.get(scene)
            if part is None:
                raise ValueError(f"episode scene absent from split: {scene}")
            if any(bool(info["task_success"]) for info in infos):
                continue
            all_failure[part].add(eid)
            for anchor in (3, 6):
                active = [info for info in infos
                          if len(info["gen_traj"]) > anchor]
                for left, right in itertools.combinations(active, 2):
                    lturns, rturns = left["gen_traj"], right["gen_traj"]
                    lfuture = sum(float(turn["oracle_turn_progress"])
                                  for turn in lturns[anchor:])
                    rfuture = sum(float(turn["oracle_turn_progress"])
                                  for turn in rturns[anchor:])
                    gap = lfuture - rfuture
                    if not math.isfinite(gap):
                        raise ValueError("nonfinite future return")
                    if abs(gap) < .25:
                        continue
                    lpast = sum(float(turn["oracle_turn_progress"])
                                for turn in lturns[:anchor])
                    rpast = sum(float(turn["oracle_turn_progress"])
                                for turn in rturns[:anchor])
                    pair = {
                        "scene": scene, "episode_id": eid,
                        "future_gap": gap,
                        "forward_prefix": point(
                            commanded_forward_meters(lturns, anchor),
                            commanded_forward_meters(rturns, anchor), gap),
                        "oracle_past_progress": point(lpast, rpast, gap),
                    }
                    rows[part][anchor].append(pair)
                    if include_same_mode and \
                            left["end_reason"] == right["end_reason"]:
                        same_mode_rows[part][anchor].append(pair)
    if steps != list(range(1, expected_steps + 1)) or \
            len(seen_episodes) != expected_episodes:
        raise ValueError("incomplete group-four source")
    report = {
        "schema": "group_future_advantage_preflight_v1",
        "source_sha256": source,
        "definition": "All-failure n=4 same-episode route pairs, both histories active past anchor, absolute future oracle return gap >=0.25; inputs stop at anchor",
        "parts": {part: {
            "all_failure_episode_groups": len(all_failure[part]),
            "anchors": {str(anchor): summarize(rows[part][anchor])
                        for anchor in (3, 6)},
        } for part in split},
        "interpretation": "Train-scene label inventory only; no model, learned reward, or navigation gain",
    }
    if include_same_mode:
        for part in split:
            report["parts"][part]["same_terminal_mode_anchors"] = {
                str(anchor): summarize(same_mode_rows[part][anchor])
                for anchor in (3, 6)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("rollout", "dataset", "scene-split", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.rollout, args.dataset, args.scene_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
