"""Freeze same-scene, different-goal instructions for temporal reward audit.

Selection uses only train metadata and hashes, before counterfactual model
scores are computed. Goal coordinates choose hard negatives but are not
provided to the representation or any proposed online reward.
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(ep["episode_id"]): ep for ep in episodes}
    if len(by_id) != len(episodes) or digest(DATASET) != source["train_sha256"]:
        raise ValueError("train source mismatch")
    by_scene = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene_id"]].append(episode)
    selected = []
    for pair in source["pairs"]:
        if pair["split"] != "audit":
            continue
        original = by_id[str(pair["episode_id"])]
        goal = original["goals"][0]["position"]
        candidates = []
        for other in by_scene[pair["scene_id"]]:
            if str(other["episode_id"]) == str(pair["episode_id"]):
                continue
            instruction = other["instruction"]["instruction_text"].strip()
            separation = distance(goal, other["goals"][0]["position"])
            if separation >= 4.0 and instruction != pair["instruction"].strip():
                candidates.append((other, separation))
        if not candidates:
            raise ValueError(f"no different-goal negative {pair['pair_id']}")
        other, separation = min(candidates, key=lambda item: hashlib.sha256(
            ("temporal-swap-v1:" + pair["pair_id"] + ":" +
             str(item[0]["episode_id"])).encode()).hexdigest())
        selected.append({"pair_id": pair["pair_id"],
                         "scene_id": pair["scene_id"],
                         "source_episode_id": pair["episode_id"],
                         "swapped_episode_id": other["episode_id"],
                         "swapped_instruction": other["instruction"]["instruction_text"].strip(),
                         "goal_euclidean_separation_m_for_selection_only": separation})
    if len(selected) != source["counts"]["audit"]:
        raise ValueError("audit swap coverage mismatch")
    output = {"schema": "temporal_instruction_swaps_v1",
              "source_manifest_sha256": digest(args.manifest),
              "train_sha256": digest(DATASET),
              "selection": "same train scene, different goal at least 4 m Euclidean; SHA256 ranked",
              "pairs": sorted(selected, key=lambda row: row["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"pairs": len(selected),
                      "min_separation_m": min(row["goal_euclidean_separation_m_for_selection_only"]
                                              for row in selected)}, indent=2))


if __name__ == "__main__":
    main()
