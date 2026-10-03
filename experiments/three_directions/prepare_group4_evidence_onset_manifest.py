"""Freeze same-start expert branches with visually evidenced goal divergence.

The source records and their image bytes select examples. No model score,
development outcome, model-audit state, or val-unseen result enters selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    source = json.loads(args.expert_manifest.read_text())
    if source["schema"] != "group4_joint_value_expert_manifest_v1":
        raise ValueError("incorrect source manifest")
    selected = {}
    inventory = {}
    seen_scenes = set()
    hash_cache = {}

    def frame_hash(path: Path) -> str:
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = digest(path)
        return hash_cache[key]

    for partition in ("fit", "development"):
        candidates = {}
        exclusions = Counter()
        for row in source["selected"][partition]:
            left_id = str(row["episode_id"])
            right_id = str(row["wrong_episode_id"])
            unordered = tuple(sorted((left_id, right_id)))
            if unordered in candidates:
                exclusions["reciprocal_duplicate"] += 1
                continue
            root = args.expert_root / row["source_part"]
            pa = root / "records" / f"{left_id}.json"
            pb = root / "records" / f"{right_id}.json"
            if digest(pa) != row["record_sha256"]:
                raise ValueError(f"source record changed {left_id}")
            if not pb.exists():
                exclusions["wrong_record_unavailable"] += 1
                continue
            a, b = json.loads(pa.read_text()), json.loads(pb.read_text())
            if a["episode_id"] != left_id or b["episode_id"] != right_id or \
                    a["scene_id"] != row["scene_id"] or \
                    b["scene_id"] != row["scene_id"] or \
                    a["wrong_instruction"].strip() != b["instruction"].strip():
                raise ValueError(f"wrong-goal record mismatch {left_id}/{right_id}")
            if a["instruction"].strip() == b["instruction"].strip():
                exclusions["same_instruction"] += 1
                continue
            initial_sha = frame_hash(root / a["initial_image"])
            if initial_sha != frame_hash(root / b["initial_image"]):
                raise ValueError(f"initial RGB mismatch {left_id}/{right_id}")
            history_3_same = len(a["turns"]) >= 3 and len(b["turns"]) >= 3 and \
                all(frame_hash(root / a["turns"][i]["image"]) ==
                    frame_hash(root / b["turns"][i]["image"]) for i in range(3))
            available = []
            for anchor in (3, 6):
                if a["turn_count"] <= anchor or b["turn_count"] <= anchor:
                    continue
                frame_a = frame_hash(root / a["turns"][anchor - 1]["image"])
                frame_b = frame_hash(root / b["turns"][anchor - 1]["image"])
                if frame_a != frame_b:
                    available.append((anchor, frame_a, frame_b))
            if not available:
                exclusions["no_preterminal_rgb_divergence"] += 1
                continue
            # Prefer turn six where the navigation consequences are visible.
            anchor, frame_a, frame_b = available[-1]
            pair = {
                "episode_a": left_id, "episode_b": right_id,
                "scene_id": row["scene_id"], "anchor": anchor,
                "shared_rgb_through_3": history_3_same,
                "evidence_onset_eligible": history_3_same and anchor == 6,
                "source_part_a": row["source_part"],
                "source_part_b": row["source_part"],
                "record_a_sha256": row["record_sha256"],
                "record_b_sha256": digest(pb),
                "initial_frame_sha256": initial_sha,
                "frame_a_sha256": frame_a,
                "frame_b_sha256": frame_b,
            }
            if unordered in candidates:
                old = candidates[unordered]
                if old["scene_id"] != pair["scene_id"] or \
                        old["anchor"] != pair["anchor"]:
                    raise ValueError(f"inconsistent reciprocal pair {unordered}")
                exclusions["reciprocal_duplicate"] += 1
                continue
            candidates[unordered] = pair
        rows = sorted(candidates.values(), key=lambda r: (
            r["scene_id"], r["episode_a"], r["episode_b"]))
        scenes = {row["scene_id"] for row in rows}
        if not scenes or scenes & seen_scenes:
            raise ValueError(f"empty or overlapping scene split {partition}")
        seen_scenes |= scenes
        selected[partition] = rows
        inventory[partition] = {
            "crossed_pairs": len(rows), "scenes": len(scenes),
            "by_anchor": {str(a): sum(r["anchor"] == a for r in rows)
                          for a in (3, 6)},
            "evidence_onset_eligible": sum(r["evidence_onset_eligible"] for r in rows),
            "exclusions": dict(exclusions),
        }
    result = {
        "schema": "group4_evidence_onset_manifest_v1",
        "selection": "source-record same-start natural different-goal expert pairs, unique unordered episode pair, both preterminal at chosen anchor, exact current-frame RGB divergence; prefer anchor six, else three; onset eligible if all first three RGBs match",
        "source_expert_manifest_sha256": digest(args.expert_manifest),
        "inventory": inventory, "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": inventory,
                      "manifest_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
