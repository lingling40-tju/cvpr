"""Compare frozen initial and adapted 2x2 development matching scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


MARGINS = ("row_a", "row_b", "column_a", "column_b")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shards(paths: list[Path], kind: str, manifest: dict,
                manifest_sha: str, weights_sha: str) -> tuple[dict, list[str]]:
    if len(paths) != 3:
        raise ValueError("expected three diagnostic shards")
    records = {}
    checkpoint_sha = None
    seen_shards = set()
    for path in paths:
        part = json.loads(path.read_text())
        shard = part["shard"]
        if part["schema"] != "group4_crossed_dev_score_shard_v1" or \
                part["source_kind"] != kind or \
                part["shards"] != 3 or shard in seen_shards or \
                part["crossed_manifest_sha256"] != manifest_sha or \
                part["frozen_weights_sha256"] != weights_sha or \
                part["pairs"] != len(part["rows"]):
            raise ValueError(f"invalid {kind} diagnostic shard {path}")
        seen_shards.add(shard)
        if checkpoint_sha is None:
            checkpoint_sha = part["checkpoint_sha256"]
        elif checkpoint_sha != part["checkpoint_sha256"]:
            raise ValueError("mixed diagnostic checkpoints")
        expected = manifest["selected"]["development"][shard::3]
        if len(expected) != len(part["rows"]):
            raise ValueError("wrong shard size")
        for row, source in zip(part["rows"], expected):
            identity = (row["episode_a"], row["episode_b"])
            if identity != (source["episode_a"], source["episode_b"]) or \
                    row["scene_id"] != source["scene_id"] or \
                    identity in records or \
                    any(not isinstance(row[name], (int, float)) or
                        not math.isfinite(row[name]) for name in MARGINS):
                raise ValueError(f"invalid diagnostic row {identity}")
            records[identity] = row
    if seen_shards != {0, 1, 2} or len(records) != 56:
        raise ValueError("incomplete 2x2 development coverage")
    return records, [digest(path) for path in paths]


def metrics(rows: dict) -> dict:
    all_rows = list(rows.values())
    by_scene = defaultdict(list)
    for row in all_rows:
        by_scene[row["scene_id"]].append(row)
    def accuracy(names: tuple[str, ...]) -> float:
        return sum(row[name] > 0 for row in all_rows for name in names) / \
               (len(all_rows) * len(names))
    return {"pairs": len(rows), "scenes": len(by_scene),
            "row_accuracy": accuracy(MARGINS[:2]),
            "column_accuracy": accuracy(MARGINS[2:]),
            "all_four_accuracy": accuracy(MARGINS),
            "both_rows_correct_pairs": sum(row["row_a"] > 0 and
                                           row["row_b"] > 0 for row in all_rows),
            "both_columns_correct_pairs": sum(row["column_a"] > 0 and
                                              row["column_b"] > 0 for row in all_rows),
            "all_four_correct_pairs": sum(all(row[name] > 0 for name in MARGINS)
                                          for row in all_rows),
            "scene_macro_all_four_accuracy": sum(
                sum(row[name] > 0 for row in scene_rows for name in MARGINS) /
                (len(scene_rows) * 4) for scene_rows in by_scene.values()) /
                len(by_scene)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--old-shard", type=Path, action="append", required=True)
    parser.add_argument("--adapted-shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "group4_crossed_prefix_manifest_v1":
        raise ValueError("wrong crossed manifest")
    manifest_sha, weights_sha = digest(args.manifest), digest(args.frozen_weights)
    old, old_hashes = load_shards(args.old_shard, "initial", manifest,
                                  manifest_sha, weights_sha)
    adapted, new_hashes = load_shards(args.adapted_shard, "adapted", manifest,
                                      manifest_sha, weights_sha)
    if set(old) != set(adapted):
        raise ValueError("unpaired crossed diagnostic")
    old_metrics, new_metrics = metrics(old), metrics(adapted)
    by_scene = defaultdict(list)
    for identity in old:
        by_scene[old[identity]["scene_id"]].append(identity)
    rng = random.Random(11)
    scenes = sorted(by_scene)
    samples = []
    for _ in range(5000):
        sampled = [rng.choice(scenes) for _ in scenes]
        ids = [identity for scene in sampled for identity in by_scene[scene]]
        n = len(ids) * 4
        samples.append(sum((adapted[i][name] > 0) - (old[i][name] > 0)
                           for i in ids for name in MARGINS) / n)
    samples.sort()
    result = {"schema": "group4_crossed_dev_diagnostic_v1",
              "interpretation": "development diagnostic with repeated research-wide scene exposure; not an independent audit or navigation gain",
              "source_sha256": {"manifest": manifest_sha,
                                "frozen_weights": weights_sha,
                                "old_shards": old_hashes,
                                "adapted_shards": new_hashes},
              "initial": old_metrics, "adapted": new_metrics,
              "paired_all_four_accuracy_change":
                  new_metrics["all_four_accuracy"] - old_metrics["all_four_accuracy"],
              "scene_cluster_bootstrap_change_95":
                  [samples[125], samples[4874]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
