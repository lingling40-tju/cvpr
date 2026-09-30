"""Freeze independently assigned labels and copy only their RGB evidence."""

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    transitions = {(str(x["episode_id"]), str(x["turn"])): x
                   for x in json.loads((ROOT / "transitions.json").read_text())}
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    records = []
    with (ROOT / "annotations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            episode_id, turn = row["episode_id"], row["turn"]
            source = transitions[episode_id, turn]
            event_index = int(row["event_index"])
            event = source["events"][event_index]
            assert row["gold"] in {"Y", "N", "U"}
            stem = f"ep{episode_id}_event{event_index}_turn{turn}"
            images = {}
            for field in ("before", "after"):
                destination = output / f"{stem}_{field}.jpg"
                shutil.copyfile(ROOT / source[field], destination)
                images[field] = {
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": sha256(destination),
                }
            records.append({
                "episode_id": episode_id,
                "scene_id": source["scene_id"],
                "instruction": source["instruction"],
                "turn": int(turn),
                "event_index": event_index,
                "event": event,
                "actions": source["actions"],
                "displacement": source["displacement"],
                "vertical_delta": source["vertical_delta"],
                "images": images,
                "gold": row["gold"],
                "reason": row["reason"],
            })
    assert len(records) == 49 and len({(r["episode_id"], r["event_index"], r["turn"])
                                        for r in records}) == 49
    bundle = {
        "source_split": "R2R val_unseen",
        "selection": "manually selected transition and control turns before verifier queries",
        "annotator": "one AI assistant reviewing images and motion without verifier predictions",
        "annotation_csv_sha256": sha256(ROOT / "annotations.csv"),
        "records": records,
    }
    (ROOT / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(len(records), len({r["episode_id"] for r in records}), bundle["annotation_csv_sha256"])


if __name__ == "__main__":
    main()
