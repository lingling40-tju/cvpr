"""Freeze a fourth disjoint 256-episode val-unseen pilot screen.

Selection reads only fixed episode and scene IDs. It excludes the three
previous pilot screens and is independent of any checkpoint outputs.
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
}
SALT = "history-process-v1|"


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
    scene_ids = list(map(str, sources["full"]["scene_ids"]))
    if len(full_ids) != 1839 or len(set(full_ids)) != 1839 or len(scene_ids) != 1839:
        raise ValueError("invalid full val-unseen manifest")
    excluded = set()
    for key in ("first", "second", "third"):
        ids = set(map(str, sources[key]["episode_ids"]))
        if len(ids) != 256 or excluded & ids:
            raise ValueError(f"{key} not unique or disjoint")
        excluded.update(ids)
    if len(excluded) != 768 or not excluded <= set(full_ids):
        raise ValueError("invalid excluded coverage")
    by_scene = {}
    for episode_id, scene in zip(full_ids, scene_ids):
        if episode_id not in excluded:
            by_scene.setdefault(scene, []).append(episode_id)
    scene_order = sorted(by_scene)
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
        pool = sorted(by_scene[scene],
                      key=lambda episode_id: (hashlib.sha256(
                          (SALT + episode_id).encode()).digest(), episode_id))
        selected[scene] = pool[:quotas[scene]]
    chosen = [episode_id
              for position in range(max(map(len, selected.values())))
              for scene in scene_order
              for episode_id in selected[scene][position:position + 1]]
    if len(chosen) != 256 or len(set(chosen)) != 256 or excluded & set(chosen):
        raise ValueError("selected IDs invalid")
    scene_by_id = dict(zip(full_ids, scene_ids))
    result = {
        "split": "val_unseen",
        "selection": {
            "method": "scene_balanced_hash_round_robin",
            "version": "history-process-v1",
            "sources_sha256": EXPECTED,
            "scene_counts": {scene: len(selected[scene]) for scene in scene_order},
        },
        "episode_ids": chosen,
        "scene_ids": [scene_by_id[episode_id] for episode_id in chosen],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(chosen), "scenes": len(scene_order),
                      "manifest_sha256": digest(args.output),
                      "disjoint_previous_episode_ids": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
