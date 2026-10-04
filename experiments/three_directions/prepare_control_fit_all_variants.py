"""Bind frozen fit IDs to a completed four-rollout control source.

This prepares a label-only replay of all four variants per episode.
No intermediate geodesic labels or visual frames are inspected here.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from prepare_policy_process_manifest import valid_info


EXPECTED_IDS_SHA = "d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--control-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != EXPECTED_IDS_SHA:
        raise ValueError("frozen ID selection changed")
    ids = json.loads(args.ids.read_text())
    old = json.loads(args.old_manifest.read_text())
    if ids["schema"] != "control_exact512_policy_fit_extension_ids_v1" or \
            ids["selected_episode_ids"] != 256 or len(ids["rows"]) != 256 or \
            ids["source_sha256"]["old_policy_manifest"] != \
            digest(args.old_manifest) or \
            old["schema"] != "policy_process_train_manifest_v1" or \
            digest(args.train_dataset) != old["train_dataset_sha256"]:
        raise ValueError("source/ID selection mismatch")
    selected_ids = {str(row["episode_id"]): str(row["scene_id"])
                    for row in ids["rows"]}
    if len(selected_ids) != 256:
        raise ValueError("duplicate selected episode")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        train = {str(row["episode_id"]): row
                 for row in json.load(stream)["episodes"]}
    if any(eid not in train or str(train[eid]["scene_id"]) != scene
           for eid, scene in selected_ids.items()):
        raise ValueError("ID-only scene and R2R train mismatch")
    groups = defaultdict(list)
    steps = []
    with args.control_rollout.open() as stream:
        for line in stream:
            row = json.loads(line)
            steps.append(int(row["step"]))
            for info in row["info"]:
                eid = str(info["episode_id"])
                groups[eid].append(info)
    if steps != list(range(1, 129)) or len(groups) != 512 or \
            any(len(group) != 4 for group in groups.values()) or \
            any(eid not in groups for eid in selected_ids):
        raise ValueError("control rollout incomplete or non-group-four")
    plans = []
    for row in ids["rows"]:
        eid, scene = str(row["episode_id"]), str(row["scene_id"])
        episode = train[eid]
        for variant, info in enumerate(groups[eid]):
            distance = float(info["distance_to_goal"])
            if not valid_info(info) or \
                    info["instruction"].strip() != \
                    episode["instruction"]["instruction_text"].strip() or \
                    not math.isfinite(distance) or distance < 0:
                raise ValueError(f"invalid source rollout {eid}/{variant}")
            plans.append({
                "seed": 11, "episode_id": eid, "variant": variant,
                "scene_id": scene,
                "trajectory_id": str(episode["trajectory_id"]),
                "terminal_mode": str(info["end_reason"]),
                "terminal_distance_m_for_replay_audit_only": distance,
                "turns": sum(bool(turn.get("executed_actions"))
                             for turn in info["gen_traj"]),
            })
    if len(plans) != 1024 or len({(p["episode_id"], p["variant"])
                                  for p in plans}) != 1024:
        raise ValueError("all-variant plan incomplete")
    report = {
        "schema": "policy_process_train_manifest_v1",
        "selection": "all four variants of each frozen ID, before intermediate geodesic replay; fit-only",
        "source_id_manifest_sha256": EXPECTED_IDS_SHA,
        "scene_split_sha256": old["scene_split_sha256"],
        "train_dataset_sha256": old["train_dataset_sha256"],
        "sources": {"11": {"path": str(args.control_rollout),
                            "sha256": digest(args.control_rollout),
                            "episode_groups": 512, "rollouts": 2048}},
        "targets": {"fit": 1024, "development": 0, "audit": 0},
        "inventory": {"fit": {"episode_ids": 256, "trajectories": 1024,
                               "scenes": 38}},
        "selected": {"fit": plans, "development": [], "audit": []},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episode_ids": 256, "variants": len(plans),
                      "manifest_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
