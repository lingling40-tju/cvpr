"""Audit and compare any two models on an identical val-unseen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_direction_eval import scene_bootstrap, summarize


def load_label(root: Path, label: str, ids: list[str], shards: int) -> dict[str, dict]:
    assert (root / f"{label}.completed").exists(), label
    rows = {}
    for shard in range(shards):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::shards]
        observed = [str(x) for x in summary["episode_ids"]]
        assert len(observed) == len(set(observed)) == len(expected)
        assert set(observed) == set(expected)
        assert summary["count"] == len(expected)
        assert summary["inference_errors"] == 0
        for eid in expected:
            row = json.loads((folder / "log" / f"stats_{eid}_0.json").read_text())
            assert str(row["id"]) == eid
            assert row.get("early_stop_reason") != "inference_error"
            assert math.isfinite(float(row["spl"])) and 0 <= float(row["spl"]) <= 1
            rows[eid] = row
    assert set(rows) == set(ids)
    return rows


def compare(candidate: dict, control: dict, ids: list[str], scenes: list[str]) -> dict:
    return {
        "sr_pp": 100 * sum(bool(candidate[i]["success"]) - bool(control[i]["success"])
                           for i in ids) / len(ids),
        "spl_pp": 100 * sum(float(candidate[i]["spl"]) - float(control[i]["spl"])
                            for i in ids) / len(ids),
        "candidate_only_successes": sum(bool(candidate[i]["success"]) and
                                        not bool(control[i]["success"]) for i in ids),
        "control_only_successes": sum(bool(control[i]["success"]) and
                                      not bool(candidate[i]["success"]) for i in ids),
        "scene_cluster_bootstrap95": scene_bootstrap(candidate, control, ids, scenes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.candidate != args.control and args.shards > 0
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    assert len(ids) == len(set(ids)) == len(scenes) == args.expected_count
    candidate = load_label(args.root, args.candidate, ids, args.shards)
    control = load_label(args.root, args.control, ids, args.shards)
    result = {
        "split": "val_unseen",
        "episodes": len(ids),
        "scenes": len(set(scenes)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate": args.candidate,
        "control": args.control,
        "candidate_metrics": summarize(candidate, ids),
        "control_metrics": summarize(control, ids),
        "paired": compare(candidate, control, ids, scenes),
        "interpretation": "One decode per checkpoint and episode; scene interval is exploratory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
