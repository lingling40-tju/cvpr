"""Freeze new R2R-train fit episode IDs before reading control rollouts.

The inputs contain episode/scene identities but no policy responses,
geodesic turn labels, images, or development scores. This manifest is a
data collection plan, not a learned reward or navigation result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


SALT = "exact512-control-fit-extension-v1|"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(row: dict) -> str:
    return hashlib.sha256((SALT + row["scene_id"] + ":" +
                           str(row["episode_id"])).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--exact512-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old_manifest.read_text())
    source = json.loads(args.exact512_manifest.read_text())
    if old["schema"] != "policy_process_train_manifest_v1" or \
            old["targets"] != {"fit": 1024, "development": 320,
                               "audit": 320} or \
            source["schema"] != "qwen3_group4_exact_start_scale_dataset_v1" or \
            source["selected_rows"] != 512 or \
            len(source["rows"]) != 512:
        raise ValueError("frozen source schemas or counts changed")
    selected = old["selected"]
    scene_sets = {part: {str(row["scene_id"]) for row in selected[part]}
                  for part in ("fit", "development", "audit")}
    if any(scene_sets[left] & scene_sets[right]
           for left, right in (("fit", "development"),
                               ("fit", "audit"),
                               ("development", "audit"))):
        raise ValueError("source scene split overlaps")
    seen_ids = {str(row["episode_id"]) for part in selected
                for row in selected[part]}
    source_ids = [str(row["episode_id"]) for row in source["rows"]]
    if len(set(source_ids)) != 512:
        raise ValueError("exact512 source repeats an episode ID")
    pools = defaultdict(list)
    for row in source["rows"]:
        scene, eid = str(row["scene_id"]), str(row["episode_id"])
        if scene in scene_sets["fit"] and eid not in seen_ids:
            pools[scene].append({"seed": 11, "episode_id": eid,
                                 "scene_id": scene})
    if sum(map(len, pools.values())) != 314 or len(pools) != 38:
        raise ValueError("new fit-scene episode inventory changed")
    ordered = {scene: sorted(rows, key=rank) for scene, rows in pools.items()}
    chosen = []
    for position in range(max(map(len, ordered.values()))):
        for scene in sorted(ordered):
            if position < len(ordered[scene]) and len(chosen) < 256:
                chosen.append(ordered[scene][position])
        if len(chosen) == 256:
            break
    if len(chosen) != 256 or \
            len({row["episode_id"] for row in chosen}) != 256 or \
            len({row["scene_id"] for row in chosen}) != 38:
        raise ValueError("fit extension selection incomplete")
    report = {
        "schema": "control_exact512_policy_fit_extension_ids_v1",
        "interpretation": "ID-only fit selection before control seed-11 rollout labels; no new policy inference or representation result.",
        "selection": "SHA-ranked scene round robin over exact512 seed-11 control IDs absent from every old part, fit scenes only",
        "source_sha256": {
            "old_policy_manifest": digest(args.old_manifest),
            "exact512_dataset_manifest": digest(args.exact512_manifest),
            "exact512_parquet": source["output_parquet_sha256"],
        },
        "eligible_new_fit_scene_episodes": sum(map(len, pools.values())),
        "eligible_scenes": len(pools),
        "selected_episode_ids": 256,
        "selected_scenes": len({row["scene_id"] for row in chosen}),
        "planned_replay_trajectories": 512,
        "fit_only_sample_gate": {
            "minimum_one_meter_regressions": 100,
            "minimum_episode_ids_with_regression": 50,
        },
        "future_variant_rule": "After the complete control seed-11 n=4 rollout is audited, select one uniformly SHA-ranked variant and one distinct variant with the largest count of >=1 m geodesic regressions per episode; ties by SHA. These are fit-only labels. Reuse the same four policy trajectories; never alter development/audit IDs.",
        "rows": chosen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"eligible": report["eligible_new_fit_scene_episodes"],
                      "selected": len(chosen),
                      "scenes": report["selected_scenes"],
                      "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
