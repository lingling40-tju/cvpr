"""Exploratory paired endpoint-distance diagnosis for the branch pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.pairs.read_bytes()
    pairs = [json.loads(line) for line in raw.decode().splitlines()]
    manifest = json.loads(args.manifest.read_text())
    ids = [str(item) for item in manifest["episode_ids"]]
    scenes = manifest["scene_ids"]
    assert manifest["split"] == "val_unseen"
    assert len(ids) == len(scenes) == len(pairs) == 256
    assert len(set(ids)) == 256
    assert [(str(row["episode_id"]), row["scene_id"]) for row in pairs] == list(zip(ids, scenes))

    def delta(row: dict) -> float:
        candidate = float(row["candidate"]["distance_to_goal_m"])
        control = float(row["control"]["distance_to_goal_m"])
        assert math.isfinite(candidate) and math.isfinite(control)
        return candidate - control

    def summarize(selected: list[dict]) -> dict:
        values = [delta(row) for row in selected]
        return {
            "episodes": len(values),
            "mean_candidate_minus_control_m": statistics.mean(values),
            "median_candidate_minus_control_m": statistics.median(values),
            "candidate_more_than_0_5m_closer": sum(value < -0.5 for value in values),
            "control_more_than_0_5m_closer": sum(value > 0.5 for value in values),
        }

    scene_names = sorted(set(scenes))
    by_scene = {scene: [row for row in pairs if row["scene_id"] == scene]
                for scene in scene_names}
    rng = random.Random(20261002)
    draws = []
    for _ in range(10000):
        sampled = [row for _ in scene_names
                   for row in by_scene[rng.choice(scene_names)]]
        draws.append(statistics.mean(delta(row) for row in sampled))
    draws.sort()

    output = {
        "interpretation": "Exploratory single-seed pilot diagnosis, not a causal mechanism test.",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "split": "val_unseen",
        "overall": summarize(pairs),
        "both_failed": summarize([row for row in pairs
                                  if not row["candidate"]["success"]
                                  and not row["control"]["success"]]),
        "both_succeeded": summarize([row for row in pairs
                                     if row["candidate"]["success"]
                                     and row["control"]["success"]]),
        "candidate_only_success": summarize([row for row in pairs
                                             if row["candidate"]["success"]
                                             and not row["control"]["success"]]),
        "control_only_success": summarize([row for row in pairs
                                           if row["control"]["success"]
                                           and not row["candidate"]["success"]]),
        "overall_mean_scene_bootstrap95_m": [draws[250], draws[9750]],
        "bootstrap_scenes": len(scene_names),
        "bootstrap_draws": len(draws),
        "bootstrap_seed": 20261002,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
