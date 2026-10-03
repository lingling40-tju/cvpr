"""Build a balanced 256-row n=4 pilot with exact-start natural contrasts.

Reads the existing R2R 4,000-row parquet and licensed train metadata on
the experiment host. Emits no instruction text or RGB in the manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_qwen3_policy_route_manifest import digest, pose


def h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    table = pq.read_table(args.source_parquet)
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate R2R train episode ID")
    by_pose = defaultdict(list)
    for row in episodes:
        by_pose[pose(row)].append(row)
    candidates = defaultdict(list)
    seen = set()
    for index, item in enumerate(table.to_pylist()):
        eid = str(item["extra_info"]["episode_id"])
        if eid in seen:
            raise ValueError("duplicate source-parquet episode ID")
        seen.add(eid)
        original = by_id[eid]
        alternatives = []
        for other in by_pose[pose(original)]:
            wrong_id = str(other["episode_id"])
            if wrong_id == eid or \
                    other["instruction"]["instruction_text"].strip() == \
                    original["instruction"]["instruction_text"].strip():
                continue
            gap = math.dist(original["goals"][0]["position"],
                            other["goals"][0]["position"])
            if gap >= 4:
                alternatives.append((h(f"qwen3-policy-exact-goal-v1:{eid}:{wrong_id}"),
                                     wrong_id, gap))
        if alternatives:
            _, wrong_id, gap = min(alternatives)
            candidates[str(original["scene_id"])].append(
                (h(f"qwen3-group4-balanced-v1:{eid}"), index, eid, wrong_id, gap))
    if sum(map(len, candidates.values())) < 256 or len(candidates) < 40:
        raise ValueError("insufficient balanced exact-start candidates")
    for scene in candidates:
        candidates[scene].sort()
    scene_order = sorted(candidates, key=lambda x: h(f"qwen3-group4-scene-v1:{x}"))
    selected = []
    round_number = 0
    while len(selected) < 256:
        for scene in scene_order:
            if round_number < len(candidates[scene]):
                selected.append((scene, *candidates[scene][round_number][1:]))
                if len(selected) == 256:
                    break
        round_number += 1
    indices = [row[1] for row in selected]
    output_table = table.take(pa.array(indices, type=pa.int64()))
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, args.output_parquet)
    rows = [{"scene_id": scene, "episode_id": eid,
             "wrong_episode_id": wrong_id, "wrong_goal_gap_m": gap}
            for scene, _, eid, wrong_id, gap in selected]
    counts = Counter(row["scene_id"] for row in rows)
    report = {"schema": "qwen3_group4_exact_start_dataset_v1",
              "selection": "256 scene-round-robin SHA-ranked rows from the existing R2R 4000 parquet; exact same scene/start pose; different natural instruction; wrong goal >=4m; wrong instruction SHA-ranked",
              "source_sha256": {"source_parquet": digest(args.source_parquet),
                                "train_dataset": digest(args.train_dataset)},
              "output_parquet_sha256": digest(args.output_parquet),
              "eligible_source_rows": sum(map(len, candidates.values())),
              "selected_rows": len(rows), "scenes": len(counts),
              "min_scene_rows": min(counts.values()),
              "max_scene_rows": max(counts.values()),
              "rows": rows}
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("source_sha256", "output_parquet_sha256",
                       "eligible_source_rows", "selected_rows", "scenes",
                       "min_scene_rows", "max_scene_rows")}, indent=2))


if __name__ == "__main__":
    main()
