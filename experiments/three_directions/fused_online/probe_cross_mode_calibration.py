"""Screen a train-scene terminal-mode offset for frozen progress scores.

This uses completed group-four training rollouts only. Simulator distance is a
pairwise analysis label; neither it nor scene identity is an online reward
input. Fit scenes select one scalar STOP offset. Development and audit scenes
test whether cross-mode ranking generalizes before any policy trial.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from mode_stratified_reward import ELIGIBLE_MODES, mode_stratified_adjustments


OFFSETS = (-2.0, -1.5, -1.0, -.75, -.5, -.25, 0.0,
           .25, .5, .75, 1.0, 1.5, 2.0)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def split_for_scene(scene: str, ordered: list[str]) -> str:
    index = ordered.index(scene)
    return "development" if index < 7 else "audit" if index < 14 else "fit"


def pair_score(pair: dict, offset: float) -> float:
    near = pair["near"]
    far = pair["far"]
    return (near["z"] + offset * near["stop"] -
            far["z"] - offset * far["stop"])


def rate(pairs: list[dict], score) -> dict:
    margins = [score(pair) for pair in pairs]
    wins = sum(value > 0 for value in margins)
    ties = sum(value == 0 for value in margins)
    return {"pairs": len(pairs), "wins": wins, "ties": ties,
            "accuracy_with_half_ties": (wins + .5 * ties) / len(pairs)
            if pairs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        scenes_by_id = {str(row["episode_id"]): row["scene_id"]
                        for row in json.load(stream)["episodes"]}
    batches = [json.loads(line) for line in args.rollout.open()]
    if len(batches) != 64 or [row["step"] for row in batches] != list(range(1, 65)):
        raise ValueError("source must be a complete 64-step rollout")
    records = []
    seen_episodes = set()
    for batch in batches:
        infos = batch["info"]
        if len(infos) != 16:
            raise ValueError("expected four groups of four per step")
        _, ordinal, _ = mode_stratified_adjustments(infos, 4)
        groups = defaultdict(list)
        for index, row in enumerate(infos):
            groups[str(row["episode_id"])].append(index)
        if len(groups) != 4:
            raise ValueError("expected four matched episode groups")
        for episode_id, indices in groups.items():
            if episode_id in seen_episodes or episode_id not in scenes_by_id or \
                    len(indices) != 4:
                raise ValueError("episode grouping or train split mismatch")
            seen_episodes.add(episode_id)
            if any(infos[i]["task_success"] for i in indices):
                continue
            eligible = [i for i in indices if infos[i]["end_reason"] in ELIGIBLE_MODES]
            for i in eligible:
                row = infos[i]
                if row["fused_reward"]["status"] != "ok" or \
                        not math.isfinite(float(row["fused_reward"]["raw"])) or \
                        not math.isfinite(float(row["distance_to_goal"])):
                    raise ValueError("invalid eligible failed rollout")
            for position, first in enumerate(eligible):
                for second in eligible[position + 1:]:
                    d_first = float(infos[first]["distance_to_goal"])
                    d_second = float(infos[second]["distance_to_goal"])
                    if abs(d_first - d_second) < 1.5:
                        continue
                    near_id, far_id = (first, second) if d_first < d_second \
                                      else (second, first)
                    near, far = infos[near_id], infos[far_id]
                    records.append({"scene": scenes_by_id[episode_id],
                                    "same_mode": near["end_reason"] == far["end_reason"],
                                    "near": {"raw": float(near["fused_reward"]["raw"]),
                                             "stop": int(near["end_reason"] == ELIGIBLE_MODES[0])},
                                    "far": {"raw": float(far["fused_reward"]["raw"]),
                                            "stop": int(far["end_reason"] == ELIGIBLE_MODES[0])},
                                    "mode_ordinal_margin": ordinal[near_id] - ordinal[far_id]})
    if len(seen_episodes) != 256:
        raise ValueError("expected 256 unique group-four episodes")
    ordered_scenes = sorted({row["scene"] for row in records},
                            key=lambda scene: hashlib.sha256(
                                ("cross-mode-calibration-v1:" + scene).encode()).hexdigest())
    if len(ordered_scenes) < 28:
        raise ValueError("insufficient train scenes for scene-disjoint split")
    for record in records:
        record["split"] = split_for_scene(record["scene"], ordered_scenes)
    fit_values = [side["raw"] for row in records if row["split"] == "fit"
                  for side in (row["near"], row["far"])]
    mean = statistics.mean(fit_values)
    scale = statistics.pstdev(fit_values)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid fit-only frozen score scale")
    for row in records:
        for side in (row["near"], row["far"]):
            side["z"] = (side["raw"] - mean) / scale
    subsets = {(split, subset): [row for row in records
                                 if row["split"] == split and
                                 (subset == "all" or row["same_mode"] ==
                                  (subset == "same_mode"))]
               for split in ("fit", "development", "audit")
               for subset in ("all", "same_mode", "cross_mode")}
    fit_cross = subsets[("fit", "cross_mode")]
    if len(fit_cross) < 50:
        raise ValueError("insufficient fit-scene cross-mode pairs")
    # One predetermined scalar grid is selected using fit scenes alone.
    chosen = max(OFFSETS, key=lambda offset: (
        rate(fit_cross, lambda row: pair_score(row, offset))["accuracy_with_half_ties"],
        -abs(offset), -offset))
    comparisons = {}
    for split in ("fit", "development", "audit"):
        comparisons[split] = {}
        for subset in ("all", "same_mode", "cross_mode"):
            pairs = subsets[(split, subset)]
            comparisons[split][subset] = {
                "mode_stratified_ordinal": rate(
                    pairs, lambda row: row["mode_ordinal_margin"]),
                "raw_unstratified": rate(pairs, lambda row: pair_score(row, 0)),
                "fit_calibrated_offset": rate(
                    pairs, lambda row: pair_score(row, chosen)),
            }
    def screen_ok(split: str) -> bool:
        item = comparisons[split]["cross_mode"]
        proposed = item["fit_calibrated_offset"]
        baseline = item["mode_stratified_ordinal"]
        return proposed["pairs"] >= 40 and \
            proposed["accuracy_with_half_ties"] >= .60 and \
            proposed["accuracy_with_half_ties"] >= \
            baseline["accuracy_with_half_ties"] + .05
    report = {
        "schema": "cross_mode_offset_train_scene_screen_v1",
        "interpretation": "Training-scene diagnostic only; no online policy or val-unseen navigation claim.",
        "source_rollout_sha256": digest(args.rollout),
        "train_dataset_sha256": digest(args.train_dataset),
        "group_size": 4, "steps": 64, "episode_groups": len(seen_episodes),
        "large_gap_failed_pairs": len(records), "scene_count": len(ordered_scenes),
        "scene_split": {split: sorted(scene for scene in ordered_scenes
                                      if split_for_scene(scene, ordered_scenes) == split)
                        for split in ("fit", "development", "audit")},
        "fit_score_mean": mean, "fit_score_sd": scale,
        "fit_selected_stop_offset_in_sd_units": chosen,
        "offset_candidates_in_sd_units": list(OFFSETS),
        "comparison": comparisons,
        "predeclared_online_pilot_gate": {
            "rule": "Both development and audit need at least 40 cross-mode pairs, calibrated accuracy >=60%, and >=5 percentage points above current ordinal reward.",
            "development_pass": screen_ok("development"),
            "audit_pass": screen_ok("audit"),
            "pass": screen_ok("development") and screen_ok("audit"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_offset": chosen,
                      "gate": report["predeclared_online_pilot_gate"],
                      "cross_mode": {key: comparisons[key]["cross_mode"]
                                     for key in ("fit", "development", "audit")}},
                     indent=2))


if __name__ == "__main__":
    main()
