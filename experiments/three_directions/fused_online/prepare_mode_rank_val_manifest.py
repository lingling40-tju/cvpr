"""Freeze a third scene-balanced val-unseen screen before mode-rank inference.

Selection uses only episode and scene IDs and excludes both prior 256-episode
screens. Neither policy outputs nor evaluation metrics enter selection.
"""

import argparse
import hashlib
import json
from pathlib import Path


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
FIRST_SHA = "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46"
SECOND_SHA = "2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if [digest(args.full), digest(args.first), digest(args.second)] != [
            FULL_SHA, FIRST_SHA, SECOND_SHA]:
        raise ValueError("source manifest checksum mismatch")
    full = json.loads(args.full.read_text())
    first = json.loads(args.first.read_text())
    second = json.loads(args.second.read_text())
    ids = list(map(str, full["episode_ids"]))
    scenes = list(map(str, full["scene_ids"]))
    excluded_first = set(map(str, first["episode_ids"]))
    excluded_second = set(map(str, second["episode_ids"]))
    excluded = excluded_first | excluded_second
    if len(ids) != 1839 or len(set(ids)) != 1839 or \
            len(excluded_first) != 256 or len(excluded_second) != 256 or \
            len(excluded) != 512 or not excluded <= set(ids):
        raise ValueError("invalid full or excluded episode coverage")
    scene_by_id = dict(zip(ids, scenes))
    by_scene = {}
    for eid in ids:
        if eid not in excluded:
            by_scene.setdefault(scene_by_id[eid], []).append(eid)
    scene_order = sorted(by_scene)
    if len(scene_order) != 9:
        raise ValueError("unexpected scene coverage after exclusions")
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
        pool = sorted(by_scene[scene], key=lambda eid:
                      (hashlib.sha256(f"mode-rank-v1|{eid}".encode()).digest(), eid))
        selected[scene] = pool[:quotas[scene]]
    chosen = [eid for position in range(max(map(len, selected.values())))
              for scene in scene_order for eid in selected[scene][position:position + 1]]
    if len(chosen) != 256 or len(set(chosen)) != 256 or excluded & set(chosen):
        raise ValueError("new subset count or disjointness mismatch")
    result = {
        "split": "val_unseen",
        "selection": {
            "method": "scene_balanced_hash_round_robin",
            "version": "mode-rank-v1",
            "source_full_sha256": FULL_SHA,
            "excluded_first_256_sha256": FIRST_SHA,
            "excluded_second_256_sha256": SECOND_SHA,
            "scene_counts": {scene: len(selected[scene]) for scene in scene_order},
        },
        "episode_ids": chosen,
        "scene_ids": [scene_by_id[eid] for eid in chosen],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(chosen), "scenes": len(scene_order),
                      "sha256": digest(args.output), "disjoint_from_both": True},
                     indent=2))


if __name__ == "__main__":
    main()
