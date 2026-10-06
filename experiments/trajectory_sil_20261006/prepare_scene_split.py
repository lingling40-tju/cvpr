"""Freeze an RL-stage scene split for a possible trajectory self-imitation pilot.

The SFT initialization may already have seen R2R-train scenes. This split only
separates scenes used in subsequent RL fitting from development and reserved
screens; it is not a claim of unseen-scene generalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


SALT = "sil-scene-split-20261006-v1/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.episodes.read_bytes()
    dataset = json.loads(gzip.decompress(source_bytes))
    episodes = dataset["episodes"]
    ids = [str(item["episode_id"]) for item in episodes]
    if len(episodes) != 10819 or len(set(ids)) != len(ids):
        raise ValueError("unexpected R2R-train episode coverage")
    counts = Counter(str(item["scene_id"]) for item in episodes)
    if len(counts) != 61:
        raise ValueError("unexpected R2R-train scene coverage")

    scenes = sorted(counts, key=lambda name: hashlib.sha256(
        (SALT + name).encode("utf-8")).hexdigest())
    partitions = {
        "development": scenes[:8],
        "reserved": scenes[8:16],
        "fit": scenes[16:],
    }
    if len(set().union(*(set(x) for x in partitions.values()))) != len(counts):
        raise ValueError("scene partition coverage mismatch")
    report = {
        "schema": "trajectory_sil_rl_scene_split_feasibility_v1",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "salt": SALT,
        "total_episodes": len(episodes),
        "total_scenes": len(counts),
        "partitions": {
            name: {
                "scenes": scene_names,
                "scene_episode_counts": {scene: counts[scene] for scene in scene_names},
                "episode_count": sum(counts[scene] for scene in scene_names),
            }
            for name, scene_names in partitions.items()
        },
        "limitation": "Disjoint only for a future RL-stage fit. The common SFT initialization and prior exploratory work may have seen these R2R-train scenes; development and reserved sets are not clean final tests.",
    }
    if report["partitions"]["development"]["episode_count"] < 256 or report["partitions"]["reserved"]["episode_count"] < 256:
        raise ValueError("insufficient episode coverage for 256-episode screens")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: [len(value["scenes"]), value["episode_count"]]
                      for name, value in report["partitions"].items()}))


if __name__ == "__main__":
    main()
