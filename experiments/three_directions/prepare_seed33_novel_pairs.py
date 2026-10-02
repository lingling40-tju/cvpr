"""Freeze episode-disjoint seed-33 group-four policy pairs for new tests.

This is an additional train-split diagnostic, not val-unseen. It excludes
all 400 episode IDs used by the seed-11/22 preference dataset, and never
uses representation scores for selection. Scene overlap remains possible.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from prepare_policy_preference_manifest import DATASET, actions, chosen, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--seed33-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior_manifest.read_text())
    old_ids = {str(pair["episode_id"]) for pair in prior["pairs"]}
    if prior["selection"]["group_size"] != 4 or \
            digest(DATASET) != prior["train_sha256"]:
        raise ValueError("prior preference manifest invalid")
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        train = {str(episode["episode_id"]): episode
                 for episode in json.load(stream)["episodes"]}
    groups = defaultdict(list)
    steps = []
    with args.seed33_rollout.open() as stream:
        for line in stream:
            row = json.loads(line)
            steps.append(row["step"])
            for info in row["info"]:
                eid = str(info["episode_id"])
                if eid not in train or info["data_source"] != "r2r" or \
                        info["action_space"] != "r2r" or \
                        info["global_start_step"] != 1:
                    raise ValueError("unexpected seed33 rollout source")
                if info["instruction"].strip() != \
                        train[eid]["instruction"]["instruction_text"].strip():
                    raise ValueError(f"instruction mismatch {eid}")
                groups[eid].append(info)
    if steps != list(range(1, 129)) or len(groups) != 512 or \
            any(len(infos) != 4 for infos in groups.values()):
        raise ValueError("incomplete seed33 group-four rollout")
    pairs = []
    for eid, infos in groups.items():
        if eid in old_ids:
            continue
        positives = [(i, info) for i, info in enumerate(infos)
                     if info["task_success"] and info["distance_to_goal"] <= 3.0]
        negatives = [(i, info) for i, info in enumerate(infos)
                     if not info["task_success"] and
                     math.isfinite(info["distance_to_goal"]) and
                     info["distance_to_goal"] >= 3.5]
        if not positives or not negatives:
            continue
        pi, pos = min(positives, key=lambda row: (len(actions(row[1])), row[0]))
        ni, neg = min(negatives,
                      key=lambda row: (row[1]["distance_to_goal"],
                                       len(actions(row[1])), row[0]))
        episode = train[eid]
        pairs.append({"episode_id": int(eid), "scene_id": episode["scene_id"],
                      "instruction": pos["instruction"],
                      "pair_id": f"seed33_novel_episode{eid}",
                      "success": chosen(pos, pi, 33),
                      "failure": chosen(neg, ni, 33),
                      "split": "audit"})
    if len(pairs) < 30:
        raise ValueError(f"too few episode-disjoint pairs: {len(pairs)}")
    scenes = sorted({pair["scene_id"] for pair in pairs})
    output = {"schema": "seed33_novel_policy_preference_v1",
              "train_sha256": digest(DATASET),
              "prior_manifest_sha256": digest(args.prior_manifest),
              "selection": {"group_size": 4,
                            "optimizer_steps_per_seed": 128,
                            "failed_distance_at_least_m": 3.5,
                            "episode_disjoint_from_prior": True,
                            "scene_disjoint_from_prior": False},
              "sources": {"33": {"groups": len(groups),
                                 "rollout_path": str(args.seed33_rollout),
                                 "rollout_sha256": digest(args.seed33_rollout)}},
              "scene_split": {"audit": scenes},
              "counts": {"audit": len(pairs)},
              "pairs": sorted(pairs, key=lambda pair: pair["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"pairs": len(pairs), "scenes": len(scenes),
                      "source_groups": len(groups),
                      "overlap_with_prior": len(old_ids &
                                                {str(p["episode_id"]) for p in pairs}),
                      "by_scene": Counter(p["scene_id"] for p in pairs)}, indent=2))


if __name__ == "__main__":
    main()
