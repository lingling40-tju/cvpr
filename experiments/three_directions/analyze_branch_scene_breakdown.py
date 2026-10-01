"""Descriptive per-scene pilot breakdown from validated paired episode export."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    assert len(rows) == 256
    assert len({str(row["episode_id"]) for row in rows}) == len(rows)
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[Path(row["scene_id"]).stem].append(row)
    assert len(by_scene) == 11

    scenes = []
    for name, items in by_scene.items():
        candidate_only = sum(bool(r["candidate"]["success"]) and not bool(r["control"]["success"]) for r in items)
        control_only = sum(bool(r["control"]["success"]) and not bool(r["candidate"]["success"]) for r in items)
        scenes.append({
            "scene_id": name,
            "episodes": len(items),
            "candidate_successes": sum(bool(r["candidate"]["success"]) for r in items),
            "control_successes": sum(bool(r["control"]["success"]) for r in items),
            "candidate_only_successes": candidate_only,
            "control_only_successes": control_only,
            "paired_sr_pp": 100 * (candidate_only - control_only) / len(items),
            "paired_spl_pp": 100 * sum(float(r["candidate"]["spl"]) - float(r["control"]["spl"]) for r in items) / len(items),
        })
    scenes.sort(key=lambda row: (-row["candidate_only_successes"] + row["control_only_successes"], row["scene_id"]))
    assert sum(row["candidate_only_successes"] for row in scenes) == 16
    assert sum(row["control_only_successes"] for row in scenes) == 9
    top = scenes[:3]
    output = {
        "split": "val_unseen",
        "episodes": len(rows),
        "scenes": scenes,
        "top_three_by_post_hoc_net_success": {
            "scene_ids": [row["scene_id"] for row in top],
            "candidate_only_successes": sum(row["candidate_only_successes"] for row in top),
            "control_only_successes": sum(row["control_only_successes"] for row in top),
        },
        "interpretation": "Exploratory post-hoc scene breakdown of a single-seed 256-episode pilot; not a confirmatory subgroup result.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["top_three_by_post_hoc_net_success"], indent=2))


if __name__ == "__main__":
    main()
