"""Check complete frozen-SFT preference scoring and correlated uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    values = [(row["preferred"]["with_eos_logprob_sum"] >
               row["rejected"]["with_eos_logprob_sum"]) for row in rows]
    by_episode = defaultdict(list)
    by_scene = defaultdict(list)
    for row, value in zip(rows, values):
        by_episode[row["episode_id"]].append(int(value))
        by_scene[row["scene_id"]].append(int(value))
    def cluster_interval(clusters: dict, seed: int) -> list[float]:
        rng = random.Random(seed)
        groups = list(clusters.values())
        samples = []
        for _ in range(5000):
            chosen = [rng.choice(groups) for _ in groups]
            samples.append(sum(sum(group) for group in chosen) /
                           sum(len(group) for group in chosen))
        samples.sort()
        return [samples[125], samples[4874]]
    return {"groups": len(rows), "correct": sum(values),
            "accuracy": sum(values) / len(values),
            "unique_episodes": len(by_episode), "scenes": len(by_scene),
            "episode_macro_accuracy": sum(sum(group) / len(group) for group in
                                          by_episode.values()) / len(by_episode),
            "episode_cluster_bootstrap_95": cluster_interval(by_episode, 11),
            "scene_cluster_bootstrap_95": cluster_interval(by_scene, 22),
            "preferred_rejected_equal_action_token_count": sum(
                row["preferred"]["action_token_count"] ==
                row["rejected"]["action_token_count"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = {"schema": "group4_first_action_sft_baseline_analysis_v1",
              "manifest_sha256": digest(args.manifest), "parts": {}}
    model_hash = None
    for part, count in (("fit", 4), ("development", 2)):
        rows = []
        image_path = args.root / f"group4_first_action_image_index_{part}.json"
        for shard in range(count):
            label = "dev" if part == "development" else part
            path = args.root / f"group4_first_action_sft_{label}_shard{shard}.json"
            result = json.loads(path.read_text())
            if result["manifest_sha256"] != digest(args.manifest) or \
                    result["image_index_sha256"] != digest(image_path) or \
                    result["part"] != part or result["shards"] != count or \
                    result["shard"] != shard or result["smoke_limit"]:
                raise ValueError(f"bad score shard {path}")
            if model_hash is None:
                model_hash = result["model_config_sha256"]
            if result["model_config_sha256"] != model_hash:
                raise ValueError("mixed reference models")
            rows.extend(result["rows"])
        selected = {row["group_id"]: row for row in manifest["selected"][part]}
        if len(rows) != len(selected) or {row["group_id"] for row in rows} != set(selected):
            raise ValueError(f"incomplete or duplicate {part} coverage")
        for row in rows:
            source = selected[row["group_id"]]
            if row["episode_id"] != str(source["episode_id"]) or \
                    row["scene_id"] != source["scene_id"]:
                raise ValueError(f"score identity mismatch {row['group_id']}")
        output["parts"][part] = summarize(rows)
    output["model_config_sha256"] = model_hash
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
