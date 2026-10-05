"""Fit-only feasibility audit for MP3D room labels as VLN target evidence.

This deliberately uses axis-aligned region boxes and literal terminal-clause
room nouns. It does not infer semantic truth, inspect RGB, or open reserved
development/audit scenes. Counts are source feasibility diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOM_NOUNS = {
    "a": r"bathroom", "b": r"bedroom", "c": r"closet",
    "d": r"dining room", "e": r"entryway|foyer",
    "f": r"family room", "h": r"hallway|hall", "k": r"kitchen",
    "l": r"living room", "o": r"office",
    "p": r"porch|terrace|deck", "s": r"stair(?:s|case)?",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regions(path: Path) -> tuple[list[tuple], int]:
    boxes, objects = [], 0
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "R":
            boxes.append((words[5], tuple(map(float, words[9:12])),
                          tuple(map(float, words[12:15]))))
        elif words[0] == "O":
            objects += 1
    return boxes, objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    if (capture["schema"] != "multiview_event_rgb_capture_manifest_v1" or
            capture["group_size"] != 4 or
            len(capture["scene_split"]["fit"]) != 38):
        raise ValueError("unexpected frozen fit source")
    episodes = json.load(gzip.open(args.train_dataset, "rt"))["episodes"]
    indexed = {str(x["episode_id"]): x for x in episodes}
    if len(indexed) != len(episodes):
        raise ValueError("duplicate R2R episode id")
    goal_fields = {field for episode in episodes for goal in episode["goals"]
                   for field in goal}
    if goal_fields != {"position", "radius"}:
        raise ValueError("R2R goal schema unexpectedly contains additional fields")
    scene_boxes, object_count, region_count = {}, 0, 0
    paths = []
    for scene in capture["scene_split"]["fit"]:
        scan = scene.split("/")[1]
        path = args.scene_root / "mp3d" / scan / f"{scan}.house"
        boxes, objects = regions(path)
        scene_boxes[scene] = boxes
        object_count += objects
        region_count += len(boxes)
        paths.append((str(path.relative_to(args.scene_root)), digest(path)))
    counts = Counter()
    ids = defaultdict(set)
    for plan in capture["selected"]["fit"]:
        eid = str(plan["episode_id"])
        episode = indexed[eid]
        if (episode["scene_id"] != plan["scene_id"] or
                episode["instruction"]["instruction_text"].strip() !=
                plan["instruction"].strip()):
            raise ValueError(f"source episode mismatch: {eid}")
        counts["fit_records"] += 1
        ids["fit_records"].add(eid)
        x, y, z = episode["goals"][0]["position"]
        # MP3D house z-up versus Habitat y-up: exploratory axis swap only.
        position = (x, z, y)
        hits = [box[0] for box in scene_boxes[plan["scene_id"]]
                if all(lo <= value <= hi for lo, value, hi in
                       zip(box[1], position, box[2]))]
        counts[f"goal_region_aabb_hits_{min(len(hits), 2)}"] += 1
        target = [category for category, regex in ROOM_NOUNS.items()
                  if re.search(r"\b(?:" + regex + r")\b",
                               plan["terminal_clause"], re.I)]
        if len(target) != 1:
            counts["not_one_literal_terminal_room_noun"] += 1
            continue
        counts["one_literal_terminal_room_noun"] += 1
        ids["one_literal_terminal_room_noun"].add(eid)
        if len(hits) != 1:
            counts["room_noun_nonunique_region_aabb"] += 1
            continue
        counts["room_noun_one_region_aabb"] += 1
        ids["room_noun_one_region_aabb"].add(eid)
        key = "room_noun_region_aabb_match" if hits[0] == target[0] else \
              "room_noun_region_aabb_mismatch"
        counts[key] += 1
        ids[key].add(eid)
    file_manifest = hashlib.sha256(json.dumps(
        paths, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {
        "schema": "mp3d_semantic_link_fit_preflight_v1",
        "capture_manifest_sha256": digest(args.capture),
        "r2r_train_json_gz_sha256": digest(args.train_dataset),
        "fit_house_file_manifest_sha256": file_manifest,
        "fit_scenes": len(scene_boxes),
        "house_region_annotations": region_count,
        "house_object_annotations": object_count,
        "counts": dict(sorted(counts.items())),
        "unique_episode_ids": {k: len(v) for k, v in sorted(ids.items())},
        "r2r_episode_fields": sorted(episodes[0]),
        "r2r_goal_fields": sorted(goal_fields),
        "source_semantic_target_id_present": bool(goal_fields -
                                                    {"position", "radius"}),
        "room_noun_rule": ROOM_NOUNS,
        "position_rule": "Exploratory Habitat (x,y,z) to MP3D house (x,z,y) and axis-aligned region box containment; no polygon or floor adjudication",
        "interpretation": "Raw human scan-region/object annotations exist, but R2R goals have no room/object ID. Literal target nouns and AABB checks have limited coverage and cannot validate ordered instruction-event semantics.",
        "development_scenes_opened": False,
        "reserved_audit_scenes_opened": False,
        "rgb_files_opened": 0,
        "navigation_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
