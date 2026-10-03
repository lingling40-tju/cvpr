"""Screen whether frozen-score separation predicts reliable failure ranks.

Uses the completed failure-only policy's train-split group-four rollouts.
Simulator terminal distance is a diagnostic label only. The one-dimensional
score-margin threshold is selected on fit scenes, then held fixed on
development and audit scenes. This is exploratory after earlier diagnostics
of overlapping training data and makes no navigation claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from mode_stratified_reward import ELIGIBLE_MODES


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(pairs: list[dict], threshold: float) -> dict:
    retained = [pair for pair in pairs if pair["margin"] >= threshold]
    wins = sum(pair["correct"] for pair in retained)
    return {"available_pairs": len(pairs), "retained_pairs": len(retained),
            "wins": wins, "accuracy": wins / len(retained) if retained else None,
            "coverage": len(retained) / len(pairs) if pairs else None,
            "net_correct_fraction_of_all_pairs":
                (2 * wins - len(retained)) / len(pairs) if pairs else None}


def active_fraction(groups: list[dict], threshold: float) -> dict:
    active = sum(any(margin >= threshold for margin in group["margins"])
                 for group in groups)
    return {"all_failure_groups": len(groups), "active_groups": active,
            "fraction": active / len(groups) if groups else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        scene_by_id = {str(row["episode_id"]): row["scene_id"]
                       for row in json.load(stream)["episodes"]}
    pairs, groups = [], []
    seen = set()
    steps = 0
    for line in args.rollout.open():
        batch = json.loads(line)
        steps += 1
        if batch["step"] != steps or len(batch["info"]) != 16:
            raise ValueError("bad sequential group-four rollout batch")
        grouped = defaultdict(list)
        for item in batch["info"]:
            grouped[str(item["episode_id"])].append(item)
        if len(grouped) != 4:
            raise ValueError("expected four episode groups per step")
        for episode_id, items in grouped.items():
            if episode_id in seen or episode_id not in scene_by_id or len(items) != 4:
                raise ValueError("duplicate or nontraining episode, or not group four")
            seen.add(episode_id)
            if any(item["task_success"] for item in items):
                continue
            scene = scene_by_id[episode_id]
            margins = []
            for mode in ELIGIBLE_MODES:
                members = [item for item in items if item["end_reason"] == mode]
                for item in members:
                    if item["fused_reward"]["status"] != "ok" or \
                            not math.isfinite(float(item["fused_reward"]["raw"])) or \
                            not math.isfinite(float(item["distance_to_goal"])):
                        raise ValueError("invalid scored failure")
                for offset, first in enumerate(members):
                    for second in members[offset + 1:]:
                        score_delta = (float(first["fused_reward"]["raw"]) -
                                       float(second["fused_reward"]["raw"]))
                        margins.append(abs(score_delta))
                        distance_delta = (float(first["distance_to_goal"]) -
                                          float(second["distance_to_goal"]))
                        if abs(distance_delta) < 1.5 or score_delta == 0:
                            continue
                        pairs.append({"scene": scene, "margin": abs(score_delta),
                                      "correct": score_delta * distance_delta < 0})
            groups.append({"scene": scene, "margins": margins})
    if steps != 64 or len(seen) != 256:
        raise ValueError("expected complete 64-step, 256-group source")
    ordered_scenes = sorted({pair["scene"] for pair in pairs},
                            key=lambda scene: hashlib.sha256(
                                ("within-mode-margin-v1:" + scene).encode()).hexdigest())
    if len(ordered_scenes) < 28:
        raise ValueError("too few scenes with comparable failed pairs")
    development, audit = set(ordered_scenes[:7]), set(ordered_scenes[7:14])
    split = lambda scene: ("development" if scene in development else
                           "audit" if scene in audit else "fit")
    pair_parts = {name: [pair for pair in pairs if split(pair["scene"]) == name]
                  for name in ("fit", "development", "audit")}
    group_parts = {name: [group for group in groups if split(group["scene"]) == name]
                   for name in ("fit", "development", "audit")}
    ordered_margins = sorted(pair["margin"] for pair in pair_parts["fit"])
    if len(ordered_margins) < 100:
        raise ValueError("underpowered fit subset")
    thresholds = [-1.0] + [ordered_margins[int((len(ordered_margins)-1) * q)]
                           for q in (.25, .50, .75)]
    fit_candidates = [summarize(pair_parts["fit"], threshold)
                      for threshold in thresholds]
    eligible = [index for index, item in enumerate(fit_candidates)
                if item["coverage"] >= .30]
    chosen_index = max(eligible, key=lambda index: (
        fit_candidates[index]["net_correct_fraction_of_all_pairs"], -index))
    threshold = thresholds[chosen_index]
    results = {}
    for name in ("fit", "development", "audit"):
        results[name] = {"ungated": summarize(pair_parts[name], -1.0),
                         "selected": summarize(pair_parts[name], threshold),
                         "active_all_failure_groups": active_fraction(
                             group_parts[name], threshold)}
    def passes(name: str) -> bool:
        result = results[name]
        selected, original = result["selected"], result["ungated"]
        active = result["active_all_failure_groups"]
        return selected["retained_pairs"] >= 20 and \
            selected["accuracy"] >= .70 and \
            selected["accuracy"] >= original["accuracy"] + .05 and \
            active["fraction"] is not None and active["fraction"] >= .50
    report = {
        "schema": "within_mode_margin_train_scene_screen_v1",
        "interpretation": "Exploratory train-scene screen after related data were inspected; no val-unseen or retrained-policy result.",
        "rollout_sha256": digest(args.rollout),
        "train_dataset_sha256": digest(args.train_dataset),
        "group_size": 4, "steps": steps, "episode_groups": len(seen),
        "all_failure_groups": len(groups), "comparable_same_mode_pairs": len(pairs),
        "scene_count": len(ordered_scenes),
        "scene_split": {"fit": sorted(set(ordered_scenes) - development - audit),
                        "development": sorted(development), "audit": sorted(audit)},
        "fit_margin_threshold_candidates": thresholds,
        "fit_candidate_metrics": fit_candidates,
        "fit_selected_index": chosen_index,
        "fit_selected_margin_threshold": threshold,
        "held_out": results,
        "predeclared_online_pilot_gate": {
            "rule": "Each held-out split must retain at least 20 pairs, reach 70% correct with a 5-point gain over ungated ranking, and activate at least half of all-failure groups.",
            "development_pass": passes("development"),
            "audit_pass": passes("audit"),
            "pass": passes("development") and passes("audit")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_threshold": threshold,
                      "gate": report["predeclared_online_pilot_gate"],
                      "held_out": {key: results[key]
                                   for key in ("development", "audit")}},
                     indent=2))


if __name__ == "__main__":
    main()
