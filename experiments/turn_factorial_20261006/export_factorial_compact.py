"""Export one compact row per frozen factorial episode from completed raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = "c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325"
ARMS = (
    "qwen3_exact_control_64step_seed11",
    "turn_rloo_terminal_64step_seed11",
    "dense_grpo_64step_seed11",
    "turn_rloo_64step_seed11",
)
FIELDS = ("episode_id", "success", "spl", "early_stop_reason", "distance_to_goal", "path_length", "oracle_success")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_root
    if not (root / "suite.completed").exists():
        raise ValueError("factorial suite is incomplete")
    manifest_bytes = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    if manifest["split"] != "val_seen" or len(ids) != 256 or len(set(ids)) != 256 or len(scenes) != 256 or len(set(scenes)) != 50:
        raise ValueError("unexpected manifest coverage")
    for arm in ARMS:
        if not (root / f"{arm}.completed").exists():
            raise ValueError(f"incomplete arm: {arm}")
    rows = []
    for index, (episode_id, scene_id) in enumerate(zip(ids, scenes)):
        row = {"episode_id": episode_id, "scene_id": scene_id}
        for arm in ARMS:
            folder = root / arm / f"shard_{index % 4:02d}" / "log"
            raw = json.loads((folder / f"stats_{episode_id}_0.json").read_text())
            if str(raw.get("episode_id", episode_id)) != episode_id:
                raise ValueError(f"episode mismatch: {arm} {episode_id}")
            if raw.get("early_stop_reason") == "inference_error":
                raise ValueError(f"inference error: {arm} {episode_id}")
            row[arm] = {key: raw.get(key) for key in FIELDS}
        rows.append(row)
    output = {
        "schema": "turn_factorial_val_seen256_compact_v1",
        "manifest_sha256": MANIFEST_SHA256,
        "split": "val_seen",
        "rows": rows,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"exported {len(rows)} paired episodes across {len(ARMS)} arms")


if __name__ == "__main__":
    main()
