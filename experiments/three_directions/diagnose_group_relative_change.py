"""Post-hoc development-only diagnosis of group-mean score differences.

This diagnostic was devised after the scalar head failed its frozen
regression gate. It cannot be used to accept that candidate for RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fit_group_relative_head import GroupRelativeScore
from history_grounding_lora import digest, rid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (manifest["schema"] != "policy_group_relative_manifest_v1" or
            checkpoint["manifest_sha256"] != digest(args.manifest)):
        raise ValueError("manifest/checkpoint mismatch")
    model = GroupRelativeScore()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = {name: {"correct": 0, "count": 0} for name in
             ("relative_better", "relative_worse",
              "absolute_forward", "absolute_regression")}
    groups = 0
    for group in manifest["groups"]["development"]:
        variants = []
        for variant in range(4):
            record_id = rid({"seed": group["seed"],
                             "episode_id": group["episode_id"],
                             "variant": variant})
            record = json.loads((args.turn_root / "development" / "records" /
                                 f"{record_id}.json").read_text())
            cache = torch.load(args.cache_root / "development" / "records" /
                               f"{record_id}.pt", map_location="cpu",
                               weights_only=True)
            distances = {turn["original_turn_index"]:
                         turn["distance_to_goal_for_label_only"]
                         for turn in record["turns"]}
            with torch.inference_mode():
                scores = {turn: float(model(cache["hidden"][index]))
                          for index, turn in enumerate(cache["anchor_turns"])}
            variants.append((distances, scores))
        usable = False
        for before, after in ((3, 6), (6, 9), (9, 12)):
            if not all(before in d and after in d and before in s and after in s
                       for d, s in variants):
                continue
            usable = True
            for distance, score in variants:
                relative_before = sum(d[before] for d, _ in variants) / 4 - \
                                  distance[before]
                relative_after = sum(d[after] for d, _ in variants) / 4 - \
                                 distance[after]
                centered_before = score[before] - \
                    sum(s[before] for _, s in variants) / 4
                centered_after = score[after] - \
                    sum(s[after] for _, s in variants) / 4
                relative_change = relative_after - relative_before
                predicted_change = centered_after - centered_before
                absolute_change = distance[before] - distance[after]
                score_change = score[after] - score[before]
                if relative_change >= 1:
                    row = stats["relative_better"]
                    row["correct"] += int(predicted_change > 0)
                    row["count"] += 1
                elif relative_change <= -1:
                    row = stats["relative_worse"]
                    row["correct"] += int(predicted_change < 0)
                    row["count"] += 1
                if absolute_change >= 1:
                    row = stats["absolute_forward"]
                    row["correct"] += int(score_change > 0)
                    row["count"] += 1
                elif absolute_change <= -1:
                    row = stats["absolute_regression"]
                    row["correct"] += int(score_change < 0)
                    row["count"] += 1
        groups += int(usable)
    for row in stats.values():
        row["accuracy"] = row["correct"] / row["count"] if row["count"] else None
    report = {"schema": "group_relative_change_diagnostic_v1",
              "manifest_sha256": digest(args.manifest),
              "checkpoint_sha256": digest(args.checkpoint),
              "complete_quartet_groups": groups,
              "development_only_post_hoc": stats,
              "interpretation": "Exploratory failure analysis, not an acceptance gate."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
