"""Predeclared, scene-aware gate for the small full-history reward screen.

The earlier single-observation score is contextual because that prompt also
included a system message. This analysis cannot establish a causal history
ablation; any promising screen needs a matched prompt ablation and RL test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confidence(hits: list[bool], scenes: list[str]) -> list[float]:
    unique = sorted(set(scenes))
    grouped = {scene: [hit for hit, name in zip(hits, scenes) if name == scene]
               for scene in unique}
    rng = random.Random(20261003)
    draws = []
    for _ in range(5000):
        chosen = [value for _ in unique for value in grouped[rng.choice(unique)]]
        draws.append(sum(chosen) / len(chosen))
    draws.sort()
    return [draws[125], draws[4875]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--single-observation-features", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe_manifest.read_text())
    source = json.loads(args.source_manifest.read_text())
    if probe["source_manifest_sha256"] != digest(args.source_manifest):
        raise ValueError("source/probe manifest mismatch")
    old = torch.load(args.single_observation_features, map_location="cpu",
                     weights_only=False)
    if old["manifest_sha256"] != digest(args.source_manifest):
        raise ValueError("single-observation cache mismatch")
    source_index = {pair["pair_id"]: i for i, pair in enumerate(source["pairs"])}
    pairs = [pair for pair in probe["pairs"] if pair["split"] == args.split]
    margins, reference, scenes, progress = [], [], [], []
    for pair in pairs:
        scores = []
        initial = []
        for role in ("success", "failure"):
            stem = pair["pair_id"] + "_" + role
            path = args.scores_root / f"{stem}_terminal.json"
            data = json.loads(path.read_text())
            if data["record_id"] != stem or data["split"] != args.split:
                raise ValueError(f"score identity mismatch {stem}")
            scores.append(float(data["stop_margin"]))
            first = args.scores_root / f"{stem}_initial.json"
            initial.append(float(json.loads(first.read_text())["stop_margin"])
                           if first.is_file() else None)
        if any(value is not None for value in initial) and \
                any(value is None for value in initial):
            raise ValueError("partial initial pair scores")
        margins.append(scores[0] - scores[1])
        scenes.append(pair["scene_id"])
        index = source_index[pair["pair_id"]]
        reference.append(float(old["stop_margin"][2 * index, -1] -
                               old["stop_margin"][2 * index + 1, -1]))
        progress.append([scores[0] > initial[0], scores[1] > initial[1]]
                        if initial[0] is not None else None)
    if any(row is not None for row in progress) and \
            any(row is None for row in progress):
        raise ValueError("partial progress coverage")
    hits = [value > 0 for value in margins]
    ref_hits = [value > 0 for value in reference]
    accuracy = sum(hits) / len(hits)
    ref_accuracy = sum(ref_hits) / len(ref_hits)
    report = {"schema": "full_history_probe_analysis_v1", "split": args.split,
              "pairs": len(pairs), "scenes": len(set(scenes)),
              "group_size_underlying_policy_data": 4,
              "full_history_success_over_failure": accuracy,
              "full_history_scene_bootstrap95": confidence(hits, scenes),
              "previous_single_observation_reference": ref_accuracy,
              "reference_prompt_differs_in_history_and_system_message": True,
              "accuracy_difference_pp": 100 * (accuracy - ref_accuracy),
              "candidate_only_correct": sum(x and not y for x, y in zip(hits, ref_hits)),
              "reference_only_correct": sum(y and not x for x, y in zip(hits, ref_hits)),
              "per_scene": [{"scene_id": scene,
                             "pairs": sum(s == scene for s in scenes),
                             "accuracy": sum(hit for hit, s in zip(hits, scenes)
                                             if s == scene) / sum(s == scene for s in scenes)}
                            for scene in sorted(set(scenes))]}
    if args.split == "development":
        report["predeclared_screen_gate"] = {"threshold": .70,
                                              "passes": accuracy >= .70}
    else:
        report["predeclared_screen_gate"] = {
            "accuracy_at_least_0_75": accuracy >= .75,
            "reference_gain_at_least_5pp": accuracy - ref_accuracy >= .05}
    if all(row is not None for row in progress):
        success_up = sum(row[0] for row in progress) / len(progress)
        failure_up = sum(row[1] for row in progress) / len(progress)
        report["endpoint_above_start"] = {
            "success": success_up, "failure": failure_up,
            "gap_pp": 100 * (success_up - failure_up)}
        if args.split == "audit":
            report["predeclared_screen_gate"]["progress_gap_at_least_10pp"] = \
                success_up - failure_up >= .10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scene"}, indent=2))


if __name__ == "__main__":
    main()
