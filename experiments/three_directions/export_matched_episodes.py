"""Export compact paired episode metrics after exact-coverage validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_label(root: Path, label: str, ids: list[str]) -> dict[str, dict]:
    assert (root / f"{label}.completed").exists(), label
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::4]
        assert summary["count"] == len(expected)
        assert summary["inference_errors"] == 0
        assert set(map(str, summary["episode_ids"])) == set(expected)
        for eid in expected:
            row = json.loads((folder / "log" / f"stats_{eid}_0.json").read_text())
            assert str(row["id"]) == eid
            assert row.get("early_stop_reason") != "inference_error"
            rows[eid] = row
    assert set(rows) == set(ids)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", default="branch64")
    parser.add_argument("--control", default="branch_control64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    ids = [str(eid) for eid in manifest["episode_ids"]]
    scenes = manifest["scene_ids"]
    assert len(ids) == len(set(ids)) == len(scenes) == 256
    candidate = load_label(args.root, args.candidate, ids)
    control = load_label(args.root, args.control, ids)
    lines = []
    for eid, scene in zip(ids, scenes):
        pair = {"episode_id": eid, "scene_id": scene}
        for label, rows in (("candidate", candidate), ("control", control)):
            row = rows[eid]
            pair[label] = {
                "success": bool(row["success"]),
                "spl": float(row["spl"]),
                "distance_to_goal_m": float(row["distance_to_goal"]),
                "oracle_success": bool(row["oracle_success"]),
                "early_stop_reason": row.get("early_stop_reason"),
            }
        lines.append(json.dumps(pair, ensure_ascii=False))
    args.output.write_text("\n".join(lines) + "\n")
    print(f"exported {len(lines)} paired episodes to {args.output}")


if __name__ == "__main__":
    main()
