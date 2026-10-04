"""Freeze a metadata-only, scene-balanced, group-four MIA development screen.

No simulator distance or success labels enter selection. The source records
remain in the licensed remote checkout; this manifest contains only IDs and
hashes. The already used R2R-train development split is exploratory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    if source.get("schema") != "policy_process_train_manifest_v1":
        raise ValueError("unexpected source manifest")
    groups = defaultdict(list)
    for path in sorted((args.record_root / "development" / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "policy_process_turn_record_v1" or \
                record["record_id"] != path.stem or \
                record["manifest_sha256"] != digest(args.manifest):
            raise ValueError(f"invalid record: {path}")
        if len(record["turns"]) < 6:
            continue
        scene = record["scene_id"]
        episode = str(record["episode_id"])
        instruction_hash = sha(record["instruction"])
        groups[(scene, episode, instruction_hash)].append(
            {"record_id": record["record_id"], "sha256": digest(path)})
    by_scene = defaultdict(list)
    for (scene, episode, instruction_hash), records in groups.items():
        if len(records) >= 4:
            by_scene[scene].append((episode, instruction_hash, records))
    if len(by_scene) != 8:
        raise ValueError(f"expected eight development scenes, got {len(by_scene)}")
    selected = []
    for scene in sorted(by_scene):
        ranked = sorted(by_scene[scene], key=lambda x: sha(
            f"route2step-mia-screen-v1/{scene}/{x[0]}/{x[1]}"))
        if len(ranked) < 2:
            raise ValueError(f"too few group-four episodes: {scene}")
        for episode, instruction_hash, records in ranked[:2]:
            four = sorted(records, key=lambda r: sha(
                f"route2step-mia-variant-v1/{r['record_id']}"))[:4]
            selected.append({"scene_id": scene, "episode_id": episode,
                             "instruction_sha256": instruction_hash,
                             "records": four})
    result = {
        "schema": "route2step_mia_group4_screen_manifest_v1",
        "source_manifest_sha256": digest(args.manifest),
        "selection": "R2R-train reused development: two SHA-ranked episode/instruction groups per each of eight scenes, four SHA-ranked >=6-turn records per group; no distance or success labels read for selection",
        "anchors": [3, 6], "groups": len(selected),
        "queries": len(selected) * 4 * 2, "selected": selected,
        "interpretation": "Exploratory offline representation screen, not an RL or navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "queries": result["queries"],
                      "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
