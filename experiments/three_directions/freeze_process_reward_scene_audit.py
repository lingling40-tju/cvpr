"""Freeze unused R2R-train scenes for a prospective reward-model check.

Only episode IDs and scene IDs are read. The manifest contains no outcomes,
distances, observations, or model predictions. All seven unused train scenes
are retained; no result-dependent episode selection is permitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from preflight_oracle_turn_labels import EXPECTED_DATASET, EXPECTED_SPLIT, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dataset) != EXPECTED_DATASET or \
            digest(args.prior_scene_split) != EXPECTED_SPLIT:
        raise ValueError("train data or prior scene split changed")
    prior = json.loads(args.prior_scene_split.read_text())
    used = {scene for scenes in prior["scene_split"].values()
            for scene in scenes}
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["episodes"]
    all_scenes = {str(row["scene_id"]) for row in rows}
    if not used <= all_scenes:
        raise ValueError("prior split contains non-train scenes")
    unused = sorted(all_scenes - used)
    if len(all_scenes) != 61 or len(unused) != 7:
        raise ValueError("unexpected train-scene coverage")
    selected = sorted(
        ({"episode_id": str(row["episode_id"]),
          "scene_id": str(row["scene_id"])}
         for row in rows if str(row["scene_id"]) in unused),
        key=lambda row: int(row["episode_id"]),
    )
    ids = [row["episode_id"] for row in selected]
    if len(selected) != 123 or len(ids) != len(set(ids)):
        raise ValueError("unexpected audit episode coverage")
    counts = Counter(row["scene_id"] for row in selected)
    report = {
        "schema": "process_reward_prospective_train_scene_audit_v1",
        "source_sha256": {"train": EXPECTED_DATASET,
                          "prior_scene_split": EXPECTED_SPLIT},
        "scene_count": len(unused),
        "episode_count": len(selected),
        "scenes": dict(sorted(counts.items())),
        "episodes": selected,
        "interpretation": (
            "Prospective scene-disjoint train-set audit for the next reward "
            "representation. The manifest is ID-only and contains no labels "
            "or predictions. Earlier research may have used these R2R train "
            "scenes; this is not independent val-unseen evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scene_count": len(unused),
                      "episode_count": len(selected),
                      "output_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
