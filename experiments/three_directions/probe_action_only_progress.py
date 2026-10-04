"""Fixed action-text-only baseline for the action-memory progress probe.

This is a cheap leakage/control analysis on fit and scene-disjoint development
records. It reads no old audit or val-unseen examples and trains no reward
model for online use.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from history_grounding_lora import digest, load_part
from train_policy_progress_lora import fixed_subset, pair_indices


FEATURES = ("bias", "forward_25", "forward_50", "forward_75",
            "forward_other", "left_15", "left_30", "left_45",
            "right_15", "right_30", "right_45", "other_turn",
            "total_action_count")


def action_features(response: str) -> np.ndarray:
    values = {name: 0.0 for name in FEATURES}
    values["bias"] = 1.0
    actions = [value.strip().lower() for value in response.split(",")]
    if not actions or any(not value for value in actions):
        raise ValueError("empty action text")
    values["total_action_count"] = len(actions)
    for action in actions:
        match = re.fullmatch(r"move forward (\d+)cm", action)
        if match:
            length = int(match.group(1))
            key = f"forward_{length}" if length in (25, 50, 75) else "forward_other"
            values[key] += 1
            continue
        match = re.fullmatch(r"turn (left|right) (\d+) degrees", action)
        if match:
            side, degrees = match.group(1), int(match.group(2))
            key = f"{side}_{degrees}" if degrees in (15, 30, 45) else "other_turn"
            values[key] += 1
            continue
        if action != "stop":
            raise ValueError(f"unknown action form: {action}")
    return np.asarray([values[name] for name in FEATURES], dtype=np.float64)


def rows(items: list[dict]) -> list[tuple[np.ndarray, int, str, str]]:
    result = []
    for item in items:
        record = item["record"]
        pairs = pair_indices(item)
        for label, kind in ((1, "forward"), (-1, "backward"),
                            (0, "stationary")):
            for after in pairs[kind]:
                response = record["turns"][after - 1]["assistant_response"]
                result.append((action_features(response), label,
                               record["scene_id"], str(record["episode_id"])))
    return result


def fit_linear(data: list[tuple[np.ndarray, int, str, str]]) -> np.ndarray:
    x = np.stack([row[0] for row in data])
    y = np.asarray([row[1] for row in data], dtype=np.float64)
    counts = {kind: int(np.count_nonzero(y == kind)) for kind in (-1, 0, 1)}
    if any(count < 100 for count in counts.values()):
        raise ValueError("insufficient fixed fit class")
    weights = np.asarray([len(data) / (3 * counts[int(label)])
                          for label in y], dtype=np.float64)
    x_mean = np.mean(x[:, 1:], axis=0)
    x_sd = np.std(x[:, 1:], axis=0)
    x_sd[x_sd < 1e-8] = 1.0
    scaled = x.copy()
    scaled[:, 1:] = (x[:, 1:] - x_mean) / x_sd
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(scaled.T @ (weights[:, None] * scaled) + penalty,
                           scaled.T @ (weights * y))
    return np.concatenate((beta, x_mean, x_sd))


def predict(row: np.ndarray, model: np.ndarray) -> float:
    width = row.size
    beta = model[:width]
    mean = model[width:width + width - 1]
    sd = model[width + width - 1:]
    scaled = row.copy()
    scaled[1:] = (scaled[1:] - mean) / sd
    return float(scaled @ beta)


def evaluate(data: list[tuple[np.ndarray, int, str, str]],
             model: np.ndarray) -> dict:
    values = [(predict(row, model), label, scene, eid)
              for row, label, scene, eid in data]
    stationary = sorted(score for score, label, _, _ in values if label == 0)
    forward = [score for score, label, _, _ in values if label == 1]
    if not stationary or not forward or not any(label == -1 for _, label, _, _ in values):
        raise ValueError("development label class absent")
    threshold = max(0.0, stationary[min(len(stationary) - 1,
                                       math.ceil(.9 * len(stationary)) - 1)])
    accuracy = {}
    macro = {}
    episodes = {}
    for kind, label in (("forward", 1), ("backward", -1)):
        subset = [(score, scene, eid) for score, y, scene, eid in values if y == label]
        hits = [score * label > 0 for score, _, _ in subset]
        accuracy[kind] = sum(hits) / len(hits)
        by_scene = defaultdict(list)
        for hit, (_, scene, _) in zip(hits, subset):
            by_scene[scene].append(hit)
        macro[kind] = sum(sum(h) / len(h) for h in by_scene.values()) / len(by_scene)
        episodes[kind] = len({eid for _, _, eid in subset})
    return {
        "pair_counts": {"forward": len(forward), "backward": sum(y == -1 for _, y, _, _ in values),
                        "stationary": len(stationary)},
        "unique_episodes_by_class": episodes,
        "accuracy": accuracy, "scene_macro_accuracy": macro,
        "balanced_direction_accuracy": .5 * sum(accuracy.values()),
        "stationary_threshold_selected_on_development": threshold,
        "stationary_false_positive_rate": sum(score > threshold for score in stationary) / len(stationary),
        "forward_recall_at_stationary_threshold": sum(score > threshold for score in forward) / len(forward),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit = load_part("fit", *common)
    dev = load_part("development", *common)
    training = rows(fit["policy"])
    model = fit_linear(training)
    small = rows(fixed_subset(dev["policy"], 96,
                              "action-memory-policy-dev"))
    report = {
        "schema": "action_only_progress_baseline_v1",
        "interpretation": "Fixed action-text-only fit/development comparator; no audit, reward, or navigation result.",
        "features": list(FEATURES), "regularization": 1.0,
        "source_sha256": {"scene_split": digest(args.scene_split),
                          "policy_manifest": digest(args.policy_manifest)},
        "fit_rows": len(training),
        "development_small": evaluate(small, model),
        "development_full": evaluate(rows(dev["policy"]), model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fit_rows": report["fit_rows"],
                      "small": report["development_small"],
                      "full": report["development_full"]}), flush=True)


if __name__ == "__main__":
    main()
