"""Freeze matched RL training rows for a possible self-imitation comparison.

Selects only R2R-train episodes in the scene split's fit partition. Both a
future candidate and its matched control must use the same generated parquet.
This file does not launch either model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SALT = "trajectory-sil-fit512-20261006-v1/"
ROW_COUNT = 512
MIN_PER_SCENE = 2


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path)
    args = parser.parse_args()

    episode_bytes = args.episodes.read_bytes()
    split_bytes = args.scene_split.read_bytes()
    split = json.loads(split_bytes)
    if digest_bytes(episode_bytes) != split["source_sha256"]:
        raise ValueError("R2R episode source differs from scene split")
    scene_of = {str(item["episode_id"]): str(item["scene_id"])
                for item in json.loads(gzip.decompress(episode_bytes))["episodes"]}
    if len(scene_of) != 10819:
        raise ValueError("unexpected episode ID coverage")
    fit_scenes = set(split["partitions"]["fit"]["scenes"])
    table = pq.read_table(args.source_parquet)
    if table.num_rows != 4000:
        raise ValueError("unexpected source parquet row count")
    rows = table.column("extra_info").to_pylist()
    ids = [str(row["episode_id"]) for row in rows]
    if len(set(ids)) != table.num_rows or any(row["split"] != "train" for row in rows):
        raise ValueError("source parquet contains duplicate or non-train rows")
    by_scene = defaultdict(list)
    index_of = {episode_id: index for index, episode_id in enumerate(ids)}
    for episode_id in ids:
        if episode_id not in scene_of:
            raise ValueError("training row missing R2R episode metadata")
        scene = scene_of[episode_id]
        if scene in fit_scenes:
            by_scene[scene].append(episode_id)
    if set(by_scene) != fit_scenes:
        raise ValueError("source parquet does not cover every fit scene")

    selected = []
    for scene in sorted(fit_scenes):
        ranked = sorted(by_scene[scene], key=lambda episode_id: digest(SALT + episode_id))
        if len(ranked) < MIN_PER_SCENE:
            raise ValueError("fit scene has too few training rows")
        selected.extend(ranked[:MIN_PER_SCENE])
    chosen = set(selected)
    remaining = [episode_id for scene in fit_scenes for episode_id in by_scene[scene]
                 if episode_id not in chosen]
    remaining.sort(key=lambda episode_id: digest(SALT + episode_id))
    selected.extend(remaining[:ROW_COUNT - len(selected)])
    selected.sort(key=lambda episode_id: digest(SALT + "order/" + episode_id))
    if len(selected) != ROW_COUNT or len(set(selected)) != ROW_COUNT:
        raise ValueError("fit512 row selection failed")
    scenes = [scene_of[episode_id] for episode_id in selected]
    counts = Counter(scenes)
    if set(counts) != fit_scenes or min(counts.values()) < MIN_PER_SCENE:
        raise ValueError("fit512 scene coverage failed")

    manifest = {
        "schema": "trajectory_sil_matched_rl_fit512_v1",
        "source_parquet_sha256": digest_bytes(args.source_parquet.read_bytes()),
        "r2r_episode_source_sha256": digest_bytes(episode_bytes),
        "scene_split_sha256": digest_bytes(split_bytes),
        "selection_salt": SALT,
        "episode_ids": selected,
        "scene_ids": scenes,
        "scene_row_counts": dict(sorted(counts.items())),
        "limitation": "Training rows only; neither a navigation evaluation nor evidence for self-imitation benefit.",
    }
    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        subset = table.take(pa.array([index_of[episode_id] for episode_id in selected], type=pa.int64()))
        pq.write_table(subset, args.output_parquet)
        manifest["generated_parquet_sha256"] = digest_bytes(args.output_parquet.read_bytes())
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(selected), "scenes": len(counts),
                      "min_rows_per_scene": min(counts.values()),
                      "max_rows_per_scene": max(counts.values())}))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
