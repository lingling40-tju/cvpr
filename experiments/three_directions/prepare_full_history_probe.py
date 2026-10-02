"""Freeze a small, scene-disjoint policy-pair screen before reading SFT scores.

Only scenes assigned to the previous probe's fit partition are eligible. The
new development and audit partitions are deterministically selected by hash;
at most four pairs per scene bound the cost of Habitat replay and multimodal
inference. Selection never inspects outcomes beyond the fixed pair manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(prefix: str, value: str) -> str:
    return hashlib.sha256((prefix + value).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    if source["schema"] != "policy_preference_v1" or \
            source["selection"]["group_size"] != 4:
        raise ValueError("requires fixed group-four policy preference pairs")
    fit_scenes = source["scene_split"]["fit"]
    if len(fit_scenes) < 24:
        raise ValueError("not enough source fit scenes")
    ordered = sorted(fit_scenes, key=lambda s: rank("full-history-v1:", s))
    scene_split = {"development": sorted(ordered[:8]), "audit": sorted(ordered[8:16])}
    by_scene = defaultdict(list)
    for pair in source["pairs"]:
        if pair["split"] == "fit":
            by_scene[pair["scene_id"]].append(pair)
    selected = []
    for split, scenes in scene_split.items():
        for scene in scenes:
            rows = sorted(by_scene[scene],
                          key=lambda p: rank("full-history-pair-v1:", p["pair_id"]))[:4]
            selected.extend({"pair_id": p["pair_id"], "scene_id": scene,
                             "split": split} for p in rows)
    counts = Counter(row["split"] for row in selected)
    if min(counts.values()) < 24:
        raise ValueError(f"insufficient fixed pairs: {counts}")
    output = {"schema": "full_history_probe_v1",
              "source_manifest_sha256": sha256(args.manifest),
              "selection": "old-fit scenes only; SHA256 ranked scenes and pairs; up to 4 pairs per scene",
              "group_size": 4, "scene_split": scene_split,
              "counts": dict(counts), "pairs": sorted(selected, key=lambda p: p["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"counts": counts, "scenes": {k: len(v) for k, v in scene_split.items()}}, indent=2))


if __name__ == "__main__":
    main()
