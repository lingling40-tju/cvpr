"""Freeze a fifth, episode-disjoint 256-case val-unseen development screen.

Selection uses only episode and scene identifiers, never policy outcomes.
The complete val-unseen set has been examined in prior work, so this screen
is a fresh fixed development slice rather than an independent test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "full": "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e",
    "first": "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46",
    "second": "2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375",
    "third": "1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc",
    "fourth": "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c",
}
SALT = "qwen-next-v1|"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in EXPECTED:
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in EXPECTED}
    if {key: digest(path) for key, path in paths.items()} != EXPECTED:
        raise ValueError("source manifest checksum mismatch")
    sources = {key: json.loads(path.read_text()) for key, path in paths.items()}
    full_ids = list(map(str, sources["full"]["episode_ids"]))
    scenes = list(map(str, sources["full"]["scene_ids"]))
    if sources["full"]["split"] != "val_unseen" or len(full_ids) != 1839 or \
            len(set(full_ids)) != 1839 or len(scenes) != 1839:
        raise ValueError("invalid full val-unseen manifest")
    excluded = set()
    for key in ("first", "second", "third", "fourth"):
        source = sources[key]
        ids = set(map(str, source["episode_ids"]))
        if source["split"] != "val_unseen" or len(source["episode_ids"]) != 256 or \
                len(ids) != 256 or excluded & ids:
            raise ValueError(f"{key} not unique or disjoint")
        excluded.update(ids)
    if len(excluded) != 1024 or not excluded <= set(full_ids):
        raise ValueError("invalid excluded coverage")
    by_scene = {}
    for eid, scene in zip(full_ids, scenes):
        if eid not in excluded:
            by_scene.setdefault(scene, []).append(eid)
    if sum(map(len, by_scene.values())) != 815:
        raise ValueError("unexpected available episode count")
    scene_order = sorted(by_scene)
    quota = {scene: min(256 // len(scene_order), len(by_scene[scene]))
             for scene in scene_order}
    remaining = 256 - sum(quota.values())
    while remaining:
        advanced = False
        for scene in scene_order:
            if quota[scene] < len(by_scene[scene]):
                quota[scene] += 1
                remaining -= 1
                advanced = True
                if remaining == 0:
                    break
        if not advanced:
            raise ValueError("insufficient unused episodes")
    selected = {}
    for scene in scene_order:
        pool = sorted(by_scene[scene],
                      key=lambda eid: (hashlib.sha256((SALT + eid).encode()).digest(), eid))
        selected[scene] = pool[:quota[scene]]
    chosen = [eid for position in range(max(map(len, selected.values())))
              for scene in scene_order
              for eid in selected[scene][position:position + 1]]
    if len(chosen) != 256 or len(set(chosen)) != 256 or excluded & set(chosen):
        raise ValueError("new screen identity check failed")
    scene_by_id = dict(zip(full_ids, scenes))
    result = {"split": "val_unseen",
              "selection": {"method": "scene_balanced_hash_round_robin",
                            "version": "qwen-next-v1",
                            "sources_sha256": EXPECTED,
                            "excluded_previous_episode_ids": 1024,
                            "available_before_selection": 815,
                            "scene_counts": {scene: len(selected[scene]) for scene in scene_order}},
              "episode_ids": chosen,
              "scene_ids": [scene_by_id[eid] for eid in chosen]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(chosen), "scenes": len(scene_order),
                      "excluded_previous_episode_ids": len(excluded),
                      "manifest_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
