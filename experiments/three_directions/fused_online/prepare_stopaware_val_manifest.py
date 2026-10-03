"""Freeze a scene-balanced 256-episode screen disjoint from prior screens.

The choice reads only episode and scene IDs, never policy outcomes. It is
deterministic so later reward designs cannot alter the evaluation subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
OLD_SHA = "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.full) != FULL_SHA or digest(args.old) != OLD_SHA:
        raise ValueError("source manifest checksum mismatch")
    full = json.loads(args.full.read_text())
    old = json.loads(args.old.read_text())
    ids = list(map(str, full["episode_ids"]))
    scenes = list(map(str, full["scene_ids"]))
    excluded = set(map(str, old["episode_ids"]))
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839:
        raise ValueError("invalid full manifest")
    if len(excluded) != 256 or not excluded <= set(ids):
        raise ValueError("invalid old subset")
    by_scene: dict[str, list[str]] = {}
    scene_by_id = dict(zip(ids, scenes))
    for eid in ids:
        if eid not in excluded:
            by_scene.setdefault(scene_by_id[eid], []).append(eid)
    scene_order = sorted(by_scene)
    # The prior screen exhausted the smallest 18-episode scene, so a fully
    # disjoint second screen can cover only the other ten unseen scenes.
    if len(scene_order) != 10 or len(set(scenes)) != 11:
        raise ValueError("unexpected scene coverage after exclusion")
    quotas = {scene: min(256 // len(scene_order), len(by_scene[scene]))
              for scene in scene_order}
    remaining = 256 - sum(quotas.values())
    while remaining:
        advanced = False
        for scene in scene_order:
            if quotas[scene] < len(by_scene[scene]):
                quotas[scene] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            raise ValueError("insufficient disjoint episodes")
    selected = {}
    for scene in scene_order:
        quota = quotas[scene]
        pool = sorted(by_scene[scene], key=lambda eid:
                      (hashlib.sha256(f"stopaware-v1|{eid}".encode()).digest(), eid))
        if len(pool) < quota:
            raise ValueError(f"insufficient new episodes in {scene}")
        selected[scene] = pool[:quota]
    # Round-robin scene ordering spreads long scenes across four shards.
    chosen = [eid for position in range(max(map(len, selected.values())))
              for scene in scene_order for eid in selected[scene][position:position + 1]]
    if len(chosen) != 256 or len(set(chosen)) != 256 or excluded & set(chosen):
        raise ValueError("new subset count or disjointness mismatch")
    result = {"split": "val_unseen",
              "selection": {"method": "scene_balanced_hash_round_robin",
                            "version": "stopaware-v1",
                            "source_full_sha256": FULL_SHA,
                            "excluded_prior_256_sha256": OLD_SHA,
                            "scene_counts": {scene: len(selected[scene]) for scene in scene_order}},
              "episode_ids": chosen,
              "scene_ids": [scene_by_id[eid] for eid in chosen]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(chosen), "scenes": len(scene_order),
                      "sha256": digest(args.output), "disjoint_from_prior": True}, indent=2))


if __name__ == "__main__":
    main()
