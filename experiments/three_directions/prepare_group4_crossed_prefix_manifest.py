"""Freeze visually diverged, exact-same-start expert trajectory crosses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.expert_manifest.read_text())
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("wrong expert source manifest")
    selected = {}
    inventory = {}
    scenes_seen = set()
    for part in ("fit", "development"):
        by_id = {row["episode_id"]: row for row in manifest["selected"][part]}
        if len(by_id) != len(manifest["selected"][part]):
            raise ValueError(f"duplicate expert episode {part}")
        candidates = {tuple(sorted((row["episode_id"], row["wrong_episode_id"])))
                      for row in by_id.values()
                      if row["wrong_episode_id"] in by_id and
                      row["turn_count"] > 6 and
                      by_id[row["wrong_episode_id"]]["turn_count"] > 6}
        records = []
        for a, b in sorted(candidates):
            left, right = by_id[a], by_id[b]
            if left["scene_id"] != right["scene_id"] or \
                    not (left["wrong_episode_id"] == b or
                         right["wrong_episode_id"] == a):
                raise ValueError(f"invalid same-start pair {part}/{a}/{b}")
            pa = args.expert_root / left["source_part"] / "records" / f"{a}.json"
            pb = args.expert_root / right["source_part"] / "records" / f"{b}.json"
            if digest(pa) != left["record_sha256"] or \
                    digest(pb) != right["record_sha256"]:
                raise ValueError(f"changed expert record {part}/{a}/{b}")
            ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
            if ra["episode_id"] != a or rb["episode_id"] != b or \
                    ra["scene_id"] != left["scene_id"] or \
                    rb["scene_id"] != right["scene_id"] or \
                    not (ra["wrong_instruction"].strip() == rb["instruction"].strip()
                         or rb["wrong_instruction"].strip() == ra["instruction"].strip()):
                raise ValueError(f"instruction/trajectory mismatch {part}/{a}/{b}")
            root_a = args.expert_root / left["source_part"]
            root_b = args.expert_root / right["source_part"]
            initial_a = digest(root_a / ra["initial_image"])
            initial_b = digest(root_b / rb["initial_image"])
            if initial_a != initial_b:
                raise ValueError(f"different initial view {part}/{a}/{b}")
            frame_a = digest(root_a / ra["turns"][5]["image"])
            frame_b = digest(root_b / rb["turns"][5]["image"])
            if frame_a == frame_b:
                continue
            records.append({"episode_a": a, "episode_b": b,
                            "scene_id": left["scene_id"],
                            "anchor": 6,
                            "source_part_a": left["source_part"],
                            "source_part_b": right["source_part"],
                            "record_a_sha256": left["record_sha256"],
                            "record_b_sha256": right["record_sha256"],
                            "initial_frame_sha256": initial_a,
                            "frame_a_sha256": frame_a,
                            "frame_b_sha256": frame_b})
        expected = {"fit": (147, 35), "development": (56, 8)}[part]
        scenes = {row["scene_id"] for row in records}
        if (len(records), len(scenes)) != expected or scenes & scenes_seen:
            raise ValueError(f"crossed pair coverage/leakage {part}: "
                             f"{len(records)}, {len(scenes)}")
        scenes_seen.update(scenes)
        selected[part] = records
        inventory[part] = {"crossed_pairs": len(records),
                           "scenes": len(scenes),
                           "contrasted_prefix_states": 4 * len(records)}
    result = {"schema": "group4_crossed_prefix_manifest_v1",
              "selection": "expert trajectories with audited natural alternate instruction, identical initial RGB bytes, different sixth-turn RGB bytes, and both longer than six actions; label-only before model training",
              "source_expert_manifest_sha256": digest(args.expert_manifest),
              "anchor": 6, "inventory": inventory,
              "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
