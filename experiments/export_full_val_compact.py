"""Export the complete val-unseen run as compact, auditable episode metrics."""

import argparse
import csv
import json
from pathlib import Path


LABELS = ["sft"] + [f"seed{seed}_{arm}" for seed in (11, 22, 33)
                    for arm in ("control", "event")]
FIELDS = ("label", "episode_id", "scene_id", "success", "spl",
          "distance_to_goal", "path_length", "oracle_success", "early_stop_reason")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    ids = [str(value) for value in manifest["episode_ids"]]
    scenes = [str(value) for value in manifest["scene_ids"]]
    assert len(ids) == len(scenes) == 1839 and len(set(ids)) == len(ids)
    assert all((args.root / f"{label}.completed").exists() for label in LABELS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for label in LABELS:
            count = 0
            for index, (episode_id, scene_id) in enumerate(zip(ids, scenes)):
                record_path = (args.root / label / f"shard_{index % 4:02d}" / "log"
                               / f"stats_{episode_id}_0.json")
                record = json.loads(record_path.read_text())
                assert str(record["id"]) == episode_id
                assert record.get("early_stop_reason") != "inference_error"
                row = {"label": label, "episode_id": episode_id, "scene_id": scene_id}
                row.update({key: record[key] for key in FIELDS[3:]})
                writer.writerow(row)
                count += 1
            assert count == len(ids)
    print(f"Wrote {len(LABELS) * len(ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
