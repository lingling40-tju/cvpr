"""Freeze ordered instruction clauses for train-only expert trajectories.

Only source metadata and text hashes are exported; instruction text and
Matterport images remain in the licensed experiment checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SPLIT_PATTERN = re.compile(r"(?<=[.!?;])\s+|,\s+|\bthen\b", re.IGNORECASE)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clauses(instruction: str) -> list[str]:
    normalized = " ".join(instruction.split())
    if not normalized:
        raise ValueError("empty instruction")
    parts = [piece.strip(" .;,\t") for piece in SPLIT_PATTERN.split(normalized)]
    parts = [piece for piece in parts if piece]
    if not parts:
        raise ValueError("instruction has no clauses")
    if len(parts) > 8:
        parts = parts[:7] + ["; ".join(parts[7:])]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ordinal = json.loads(args.ordinal_manifest.read_text())
    if ordinal.get("version") != 1 or \
            len(ordinal["subsets"]["fit"]["episode_ids"]) != 512 or \
            len(ordinal["subsets"]["calibration"]["episode_ids"]) != 128 or \
            set(ordinal["fit_scenes"]) & set(ordinal["calibration_scenes"]):
        raise ValueError("unexpected source manifest")
    selected = {}
    summary = {}
    seen = set()
    for part in ("fit", "calibration"):
        rows = []
        for eid in ordinal["subsets"][part]["episode_ids"]:
            path = args.record_root / part / "records" / f"{eid}.json"
            source = json.loads(path.read_text())
            if source["episode_id"] != eid or source["scene_id"] not in \
                    ordinal["fit_scenes" if part == "fit" else "calibration_scenes"] or \
                    eid in seen or len(source["frames"]) != 6:
                raise ValueError(f"invalid source episode {part}/{eid}")
            seen.add(eid)
            pieces = clauses(source["instruction"])
            rows.append({"episode_id": eid, "scene_id": source["scene_id"],
                         "record_sha256": digest(path),
                         "instruction_sha256": text_hash(source["instruction"]),
                         "clause_sha256": [text_hash(p) for p in pieces],
                         "clause_count": len(pieces)})
        pairs = ordinal["subsets"][part]["pairs"]
        ids = {row["episode_id"] for row in rows}
        if any(pair["left"] not in ids or pair["right"] not in ids
               for pair in pairs):
            raise ValueError(f"pair coverage mismatch {part}")
        selected[part] = rows
        summary[part] = {
            "episodes": len(rows), "scenes": len({r["scene_id"] for r in rows}),
            "natural_pairs": len(pairs),
            "multi_clause": sum(r["clause_count"] >= 2 for r in rows),
            "clause_count_histogram": {str(n): sum(r["clause_count"] == n for r in rows)
                                       for n in range(1, 9)},
        }
    result = {
        "schema": "clause_alignment_manifest_v1",
        "selection": "all 512 fit and 128 calibration expert records from frozen train-only ordinal manifest; deterministic sentence/comma/then segmentation, at most eight clauses; no model-based selection",
        "ordinal_manifest_sha256": digest(args.ordinal_manifest),
        "summary": summary, "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
