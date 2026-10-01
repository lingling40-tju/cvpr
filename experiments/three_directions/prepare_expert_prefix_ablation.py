"""Prepare an equal-action-count expert-prefix ablation for branch training.

This is a curriculum control, not an evaluation result. It preserves episode
rows and replaces each policy replay prefix by the first equally many grouped
actions from that episode's train-split reference trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.source)
    assert len(source) in (256, 512)
    source_rows = source.to_dict("records")
    output_rows = []
    lengths = Counter()
    ids = set()
    for row in source_rows:
        new_row = copy.deepcopy(row)
        original = new_row["extra_info"]
        assert original["split"] == "train" and original["alternative_mode"] == "branch"
        episode_id = int(original["episode_id"])
        assert episode_id not in ids
        ids.add(episode_id)
        length = len(original["forced_history_actions"])
        assert 4 <= length <= 9
        expert = list(original["gt_actions"][:length])
        assert len(expert) == length and "stop" not in expert
        original["forced_history_actions"] = expert
        lengths[length] += 1
        output_rows.append(new_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=source.columns).to_parquet(args.output, index=False)
    written = pd.read_parquet(args.output).to_dict("records")
    assert len(written) == len(source_rows) == len(ids)
    for old, new in zip(source_rows, written):
        old_info, new_info = old["extra_info"], new["extra_info"]
        assert old_info["episode_id"] == new_info["episode_id"]
        assert len(old_info["forced_history_actions"]) == len(new_info["forced_history_actions"])
        assert list(new_info["forced_history_actions"]) == list(
            new_info["gt_actions"][:len(new_info["forced_history_actions"])]
        )

    diagnostics = {
        "source": str(args.source), "source_sha256": sha256(args.source),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "rows": len(written), "unique_train_episodes": len(ids),
        "prefix_action_count_distribution": dict(sorted(lengths.items())),
        "intervention": "same episode rows and grouped-action prefix count; expert rather than policy replay",
        "limitations": "Grouped actions can travel different physical distances; this is not a same-pose control.",
    }
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
