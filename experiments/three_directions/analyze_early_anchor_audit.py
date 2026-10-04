"""Paired train-scene audit of a frozen turn-3-only potential checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


MANIFEST_SHA = "dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1"
CHECKPOINT_SHA = "0b955f0cde2d77a89f48f0d72e2346c65afaf1b46bd1578d716caf4ae5ef3729"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_meters(record: dict) -> float:
    total = 0.0
    for turn in record["input"]["action_history_by_anchor"]["3"]:
        for action in turn["executed_actions"]:
            if action.startswith("move forward "):
                match = re.fullmatch(r"move forward (\d+)cm", action)
                if match is None:
                    raise ValueError(f"unrecognized forward action: {action}")
                total += int(match.group(1)) / 100.0
    return total


def macro(rows: list[dict]) -> dict:
    by_episode = defaultdict(list)
    baseline_episode = defaultdict(list)
    for row in rows:
        key = (row["scene"], row["episode"])
        by_episode[key].append(row["correct"])
        baseline_episode[key].append(row["action_baseline"])
    by_scene = defaultdict(list)
    for (scene, _), values in by_episode.items():
        by_scene[scene].append(sum(values) / len(values))
    scene_means = [sum(values) / len(values)
                   for _, values in sorted(by_scene.items())]
    interval = None
    if len(scene_means) >= 2:
        rng = random.Random(20261005)
        draws = sorted(sum(rng.choices(scene_means, k=len(scene_means))) /
                       len(scene_means) for _ in range(10000))
        interval = [draws[249], draws[9749]]
    return {
        "pairs": len(rows), "unique_episode_ids": len(by_episode),
        "scenes": len(by_scene),
        "episode_macro": (sum(sum(v) / len(v) for v in by_episode.values()) /
                          len(by_episode)) if by_episode else None,
        "scene_macro": sum(scene_means) / len(scene_means) if scene_means else None,
        "scene_bootstrap95": interval,
        "forward_action_episode_macro": (
            sum(sum(v) / len(v) for v in baseline_episode.values()) /
            len(baseline_episode)) if baseline_episode else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "rgb-root", "verification", "scores",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != MANIFEST_SHA:
        raise ValueError("frozen audit manifest changed")
    manifest = json.loads(args.manifest.read_text())
    verification = json.loads(args.verification.read_text())
    score_summary = json.loads(args.scores.with_suffix(".summary.json").read_text())
    if verification.get("schema") != "early_anchor_group4_audit_replay_verification_v1" or \
            verification.get("manifest_sha256") != MANIFEST_SHA or \
            verification.get("selected_records") != 540 or \
            score_summary.get("schema") != "early_anchor_observation_only_scores_v1" or \
            score_summary.get("manifest_sha256") != MANIFEST_SHA or \
            score_summary.get("checkpoint_sha256") != CHECKPOINT_SHA or \
            score_summary.get("scores_sha256") != digest(args.scores):
        raise ValueError("audit replay or frozen scores changed")
    scores = {}
    with args.scores.open() as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["record_id"]
            if rid in scores:
                raise ValueError("duplicate model score")
            scores[rid] = float(row["score"])
            if not math.isfinite(scores[rid]):
                raise ValueError("nonfinite model score")
    if set(scores) != {plan["record_id"] for plan in manifest["plans"]}:
        raise ValueError("score coverage mismatch")
    by_group = defaultdict(list)
    root = args.rgb_root / "audit"
    for plan in manifest["plans"]:
        rid = plan["record_id"]
        record = json.loads((root / "records" / f"{rid}.json").read_text())
        audit = json.loads((root / "audits" / f"{rid}.json").read_text())
        anchor = next(row for row in audit["turns"] if row["turn"] == 3)
        by_group[(plan["seed"], str(plan["episode_id"]))].append(
            (plan, float(anchor["after_distance_m"]), scores[rid],
             forward_meters(record)))
    rows = []
    for (_, eid), group in by_group.items():
        group.sort(key=lambda row: row[0]["variant"])
        for index, (left, left_distance, left_score, left_action) in enumerate(group):
            for right, right_distance, right_score, right_action in group[index + 1:]:
                gap = right_distance - left_distance
                if abs(gap) < 1.0:
                    continue
                sign = 1 if gap > 0 else -1
                signed_score = sign * (left_score - right_score)
                signed_action = sign * (left_action - right_action)
                rows.append({
                    "scene": left["scene_id"], "episode": eid,
                    "same_mode": left["terminal_mode"] == right["terminal_mode"],
                    "preferred_side": "left" if sign > 0 else "right",
                    "correct": (1.0 if signed_score > 0 else
                                .5 if signed_score == 0 else 0.0),
                    "action_baseline": (1.0 if signed_action > 0 else
                                        .5 if signed_action == 0 else 0.0),
                })
    if len(rows) != verification["eligible_distance_gap_ge_1m_pairs"] or \
            len({row["episode"] for row in rows}) != \
            verification["eligible_unique_episode_ids"] or \
            len({row["scene"] for row in rows}) != verification["eligible_scenes"]:
        raise ValueError("audited pair coverage changed")
    all_metrics = macro(rows)
    same_metrics = macro([row for row in rows if row["same_mode"]])
    gate = bool(
        verification["predeclared_minimum_coverage_met"] and
        all_metrics["episode_macro"] >= .75 and
        same_metrics["episode_macro"] is not None and
        same_metrics["episode_macro"] >= .70 and
        all_metrics["scene_macro"] >= .70 and
        all_metrics["episode_macro"] -
        all_metrics["forward_action_episode_macro"] >= .05)
    result = {
        "schema": "early_anchor_group4_train_scene_audit_v1",
        "manifest_sha256": MANIFEST_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "verification_sha256": digest(args.verification),
        "scores_sha256": digest(args.scores),
        "all": all_metrics, "same_terminal_mode": same_metrics,
        "different_terminal_mode": macro(
            [row for row in rows if not row["same_mode"]]),
        "by_preferred_side": {
            side: macro([row for row in rows if row["preferred_side"] == side])
            for side in ("left", "right")},
        "gate": {
            "minimum_coverage": verification["predeclared_minimum_coverage_met"],
            "all_episode_ge_75": all_metrics["episode_macro"] >= .75,
            "same_mode_episode_ge_70": same_metrics["episode_macro"] is not None and
            same_metrics["episode_macro"] >= .70,
            "all_scene_ge_70": all_metrics["scene_macro"] >= .70,
            "all_episode_vs_action_ge_5pp": all_metrics["episode_macro"] -
            all_metrics["forward_action_episode_macro"] >= .05,
            "passed": gate,
        },
        "interpretation": "Post-development train-scene audit; no instruction-swap or navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
