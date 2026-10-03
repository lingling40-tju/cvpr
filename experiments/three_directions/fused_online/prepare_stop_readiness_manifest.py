"""Freeze a train-only on-policy endpoint set for STOP-readiness diagnostics.

This uses the completed stop-aware pilot as an exploratory distribution-shift
probe. Its labels and scene identities are never taken from val-unseen.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.train_dataset, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in episodes}
    assert len(by_id) == len(episodes)
    records = []
    steps = []
    for line in args.rollout.open():
        batch = json.loads(line)
        steps.append(batch["step"])
        assert len(batch["info"]) == 16
        for index, row in enumerate(batch["info"]):
            if not row["task_success"] and row["end_reason"] != "stopped but goal not reached.":
                continue
            episode = by_id[str(row["episode_id"])]
            assert episode["instruction"]["instruction_text"].strip() == row["instruction"].strip()
            actions = [action for turn in row["gen_traj"]
                       for action in turn["executed_actions"]]
            assert actions and actions[-1] == "stop" and "stop" not in actions[:-1]
            records.append({
                "record_id": f"step{batch['step']:03d}_row{index:02d}",
                "episode_id": str(row["episode_id"]),
                "scene_id": str(episode["scene_id"]),
                "instruction": row["instruction"],
                "actions": actions,
                "task_success": bool(row["task_success"]),
                "end_reason": row["end_reason"],
                "terminal_distance_m_for_replay_audit_only": float(row["distance_to_goal"]),
            })
    assert steps == list(range(1, 65))
    assert len(records) == 531
    assert len({row["record_id"] for row in records}) == len(records)
    assert Counter(row["task_success"] for row in records) == {True: 203, False: 328}
    manifest = {
        "schema": "stop_readiness_onpolicy_train_v1",
        "interpretation": "Exploratory train-scene endpoint probe after a negative online pilot; do not treat as independent validation or navigation evidence.",
        "rollout_sha256": digest(args.rollout),
        "train_dataset_sha256": digest(args.train_dataset),
        "selection": "all successful and unsuccessful voluntary-STOP rollouts in 64-step group-four seed-11 pilot",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(records),
                      "scenes": len({row["scene_id"] for row in records}),
                      "successes": 203, "failed_stops": 328,
                      "manifest_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
