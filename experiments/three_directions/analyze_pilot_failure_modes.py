"""Summarize termination and near-miss outcomes on the frozen 256 episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_stats(path: Path, episode_id: str) -> dict:
    row = json.loads(path.read_text())
    assert str(row["id"]) == episode_id, (path, episode_id)
    assert row.get("early_stop_reason") != "inference_error", path
    return row


def summarize(rows: dict[str, dict], ids: list[str]) -> dict:
    assert set(rows) == set(ids)
    return {
        "episodes": len(ids),
        "successes": sum(bool(rows[eid]["success"]) for eid in ids),
        "oracle_successes": sum(bool(rows[eid]["oracle_success"]) for eid in ids),
        "oracle_without_task_success": sum(
            bool(rows[eid]["oracle_success"]) and not bool(rows[eid]["success"])
            for eid in ids
        ),
        "early_stop_reasons": dict(
            Counter(str(rows[eid].get("early_stop_reason")) for eid in ids)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--old-full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot_manifest = json.loads((args.pilot_root / "manifest.json").read_text())
    old_manifest = json.loads((args.old_full_root / "manifest.json").read_text())
    ids = [str(eid) for eid in pilot_manifest["episode_ids"]]
    assert len(ids) == len(set(ids)) == 256
    old_ids = [str(eid) for eid in old_manifest["episode_ids"]]
    old_indices = {eid: i for i, eid in enumerate(old_ids)}
    assert len(old_indices) == len(old_ids) and set(ids) <= set(old_indices)

    result = {
        "split": "val_unseen",
        "manifest_episode_count": len(ids),
        "notes": [
            "oracle_success is the evaluator's trajectory-level oracle metric.",
            "An oracle-success/task-failure mismatch does not prove the final pose was within 3 m.",
            "The older seed-11 control used different training rows from these candidate arms.",
        ],
        "models": {},
    }
    old_rows = {
        eid: load_stats(
            args.old_full_root / "seed11_control" / f"shard_{old_indices[eid] % 4:02d}"
            / "log" / f"stats_{eid}_0.json", eid
        )
        for eid in ids
    }
    result["models"]["seed11_control"] = summarize(old_rows, ids)
    for label in ("branch64", "recovery64", "counterfactual64"):
        assert (args.pilot_root / f"{label}.completed").exists(), label
        rows = {}
        for shard in range(4):
            folder = args.pilot_root / label / f"shard_{shard:02d}"
            summary = json.loads((folder / "summary.json").read_text())
            expected = ids[shard::4]
            assert summary["count"] == len(expected)
            assert summary["inference_errors"] == 0
            assert set(map(str, summary["episode_ids"])) == set(expected)
            for eid in expected:
                rows[eid] = load_stats(folder / "log" / f"stats_{eid}_0.json", eid)
        result["models"][label] = summarize(rows, ids)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
