"""Build a portable review package without prior labels or verifier predictions."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "item_id", "instruction", "event_type", "event_target", "source_phrase",
    "actions", "displacement_m", "vertical_delta_m", "before_image",
    "after_image", "label", "comment",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="destination .zip path")
    args = parser.parse_args()
    bundle = json.loads((ROOT / "bundle.json").read_text())
    records = list(bundle["records"])
    random.Random(20261001).shuffle(records)
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "eventtrace_blind_review"
        image_dir = package / "images"
        image_dir.mkdir(parents=True)
        rows = []
        for record in records:
            key = f"{record['episode_id']}:{record['event_index']}:{record['turn']}"
            item_id = hashlib.sha256(key.encode()).hexdigest()[:12]
            names = {}
            for side in ("before", "after"):
                image = record["images"][side]
                src = ROOT / image["path"]
                assert hashlib.sha256(src.read_bytes()).hexdigest() == image["sha256"]
                name = f"{item_id}_{side}.jpg"
                shutil.copy2(src, image_dir / name)
                names[side] = f"images/{name}"
            rows.append({
                "item_id": item_id,
                "instruction": record["instruction"],
                "event_type": record["event"]["type"],
                "event_target": record["event"]["target"],
                "source_phrase": record["event"]["source_text"],
                "actions": "; ".join(record["actions"]),
                "displacement_m": record["displacement"],
                "vertical_delta_m": record["vertical_delta"],
                "before_image": names["before"],
                "after_image": names["after"],
                "label": "",
                "comment": "",
            })
        assert len(rows) == len({row["item_id"] for row in rows})
        with (package / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (package / "INSTRUCTIONS.txt").write_text(
            "EventTrace independent visual review\n\n"
            "For each row, compare the before and after image, instruction, "
            "event phrase, executed actions, and displacement. In label, enter:\n"
            "Y: the specified event newly completed during these actions;\n"
            "N: it did not newly complete (including already completed before);\n"
            "U: available evidence cannot determine completion.\n\n"
            "For stop_near, a STOP action must have executed. For enter/pass, "
            "seeing the target without crossing or passing is N. Use U when "
            "the target or crossing is visually ambiguous. Do not consult "
            "other reviewers, existing labels, or the verifier predictions. "
            "Write a brief comment on difficult or U cases. Return labels.csv.\n",
            encoding="utf-8",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        base = args.output.with_suffix("")
        made = shutil.make_archive(str(base), "zip", root_dir=Path(tmp),
                                   base_dir=package.name)
        assert Path(made).resolve() == args.output.resolve()
        print(f"{args.output}: {len(rows)} items")


if __name__ == "__main__":
    main()
