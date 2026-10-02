"""Freeze fresh scene splits and instruction negatives for temporal v2.

New development/audit scenes come from prior fit scenes not used in the
full-history probe. Previously inspected scenes become fit-only. Every
negative instruction is from the same train scene but a goal at least 4 m
away; no navigation result or model score enters selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DATASET = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(prefix: str, value: str) -> str:
    return hashlib.sha256((prefix + value).encode()).hexdigest()


def separation(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    prior = json.loads(args.history_probe.read_text())
    if prior["source_manifest_sha256"] != digest(args.manifest) or \
            source["selection"]["group_size"] != 4 or \
            digest(DATASET) != source["train_sha256"]:
        raise ValueError("source provenance mismatch")
    seen = set(prior["scene_split"]["development"] +
               prior["scene_split"]["audit"])
    eligible = sorted(set(source["scene_split"]["fit"]) - seen,
                      key=lambda s: rank("temporal-contrastive-v2:", s))
    if len(eligible) < 25:
        raise ValueError("too few previously untested scenes")
    development, audit = set(eligible[:8]), set(eligible[8:16])
    fit = set(source["scene_split"]["fit"] +
              source["scene_split"]["development"] +
              source["scene_split"]["audit"]) - development - audit
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(ep["episode_id"]): ep for ep in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("ambiguous train episode IDs")
    by_scene = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene_id"]].append(episode)
    rows = []
    for pair in source["pairs"]:
        scene = pair["scene_id"]
        split = "audit" if scene in audit else "development" if scene in development else "fit"
        original = by_id[str(pair["episode_id"])]
        goal = original["goals"][0]["position"]
        candidates = []
        for other in by_scene[scene]:
            if str(other["episode_id"]) == str(pair["episode_id"]):
                continue
            instruction = other["instruction"]["instruction_text"].strip()
            gap = separation(goal, other["goals"][0]["position"])
            if gap >= 4.0 and instruction != pair["instruction"].strip():
                candidates.append((other, gap))
        if not candidates:
            raise ValueError(f"no instruction negative for {pair['pair_id']}")
        other, gap = min(candidates, key=lambda x: rank(
            "temporal-contrastive-negative-v2:",
            pair["pair_id"] + ":" + str(x[0]["episode_id"])))
        rows.append({"pair_id": pair["pair_id"], "scene_id": scene,
                     "split": split, "source_episode_id": pair["episode_id"],
                     "swapped_episode_id": other["episode_id"],
                     "swapped_instruction": other["instruction"]["instruction_text"].strip(),
                     "goal_separation_m_for_selection_only": gap})
    counts = Counter(row["split"] for row in rows)
    if counts["fit"] < 250 or counts["development"] < 40 or counts["audit"] < 40:
        raise ValueError(f"underpowered split {counts}")
    output = {"schema": "temporal_contrastive_v2_manifest",
              "source_manifest_sha256": digest(args.manifest),
              "full_history_probe_sha256": digest(args.history_probe),
              "train_sha256": digest(DATASET),
              "group_size": 4,
              "selection": "fresh scenes outside prior full-history probe; same-scene wrong goal >=4m; SHA256 ranking",
              "scene_split": {"fit": sorted(fit),
                              "development": sorted(development),
                              "audit": sorted(audit)},
              "counts": dict(counts), "pairs": sorted(rows, key=lambda p: p["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"counts": counts,
                      "scenes": {k: len(v) for k, v in output["scene_split"].items()},
                      "min_goal_separation_m": min(row["goal_separation_m_for_selection_only"]
                                                   for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
