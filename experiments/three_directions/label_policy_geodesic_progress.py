"""Attach train-only geodesic progress to already cached policy RGB frames.

The privileged simulator distance is used only to fit/audit a deployable
image-instruction potential. Evaluation policy rewards must not read it.
No image or model inference is repeated in this pass.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import habitat
from habitat import Env

from collect_policy_preference_frames import (
    CONFIG, DATASET, action_steps, canonical_scene, digest,
)


def label_one(env: Env, observation: dict, row: dict, role: str,
              frames_root: Path) -> dict:
    record_id = row["pair_id"] + "_" + role
    record = json.loads((frames_root / "records" / f"{record_id}.json").read_text())
    if str(env.current_episode.episode_id) != str(row["episode_id"]) or \
            canonical_scene(str(env.current_episode.scene_id)) != row["scene_id"] or \
            observation["instruction"]["text"].strip() != row["instruction"].strip():
        raise ValueError("replay episode/scene/instruction mismatch")
    if record["record_id"] != record_id or len(record["frames"]) != 4:
        raise ValueError("cached frame record mismatch")
    checkpoints = [frame["action_index"] for frame in record["frames"]]
    if checkpoints[0] != 0 or checkpoints != sorted(set(checkpoints)):
        raise ValueError("invalid sparse checkpoints")
    actions = [action for action in row[role]["executed_actions"] if action != "stop"]
    if checkpoints[-1] != len(actions):
        raise ValueError("terminal checkpoint mismatch")
    distances = [float(env.get_metrics()["distance_to_goal"])]
    for index, action in enumerate(actions, 1):
        code, repeat = action_steps(action)
        for _ in range(repeat):
            if env.episode_over:
                raise RuntimeError("replay ended early")
            observation = env.step({"action": code})
        if index in checkpoints[1:]:
            distances.append(float(env.get_metrics()["distance_to_goal"]))
    if len(distances) != 4 or \
            abs(distances[-1] - row[role]["terminal_distance_m_for_replay_audit_only"]) > .25:
        raise RuntimeError("geodesic replay drift")
    return {"schema": "policy_geodesic_progress_v1", "record_id": record_id,
            "pair_id": row["pair_id"], "role": role,
            "scene_id": row["scene_id"], "split": row["split"],
            "action_indices": checkpoints, "distance_to_goal_m": distances,
            "progress_from_start_m": [distances[0] - value for value in distances]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    collection = json.loads((args.frames_root / "summary.json").read_text())
    if digest(DATASET) != manifest["train_sha256"] or \
            collection["manifest_sha256"] != digest(args.manifest) or \
            collection["completed_trajectories"] != 2 * len(manifest["pairs"]) or \
            collection["errors"]:
        raise ValueError("source manifest or frame cache incomplete")
    pairs = manifest["pairs"][:args.limit_pairs or None]
    if not pairs or args.limit_pairs < 0:
        raise ValueError("empty/invalid pair limit")
    plans = [(row, role) for row in pairs for role in ("success", "failure")]
    from VLN_CE.vlnce_baselines.config.default import get_config
    config = get_config(CONFIG)
    config.defrost()
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "train"
    config.TASK_CONFIG.TASK.NDTW.SPLIT = "train"
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL"]
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.TASK_CONFIG.freeze()
    config.freeze()
    dataset = habitat.datasets.make_dataset(config.TASK_CONFIG.DATASET.TYPE,
                                            config=config.TASK_CONFIG.DATASET)
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    if len(by_id) != len(dataset.episodes):
        raise ValueError("duplicate train episode id")
    plans.sort(key=lambda p: (p[0]["scene_id"], p[0]["episode_id"], p[0]["pair_id"], p[1]))
    requested = defaultdict(deque)
    episodes = []
    for row, role in plans:
        eid = str(row["episode_id"])
        ep = by_id.get(eid)
        if ep is None or canonical_scene(str(ep.scene_id)) != row["scene_id"]:
            raise ValueError(f"missing episode {eid}")
        episodes.append(ep)
        requested[eid].append((row, role))
    dataset.episodes = episodes
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed, errors = 0, []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in episodes:
            observation = env.reset()
            row, role = requested[str(env.current_episode.episode_id)].popleft()
            record_id = row["pair_id"] + "_" + role
            path = args.output_root / "records" / f"{record_id}.json"
            path.parent.mkdir(exist_ok=True)
            if path.is_file():
                cached = json.loads(path.read_text())
                if cached["schema"] == "policy_geodesic_progress_v1" and \
                        cached["record_id"] == record_id and \
                        len(cached["distance_to_goal_m"]) == 4:
                    completed += 1
                    continue
            try:
                label = label_one(env, observation, row, role, args.frames_root)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(label, indent=2) + "\n")
                os.replace(temp, path)
                completed += 1
                if completed % 20 == 0:
                    print(f"labeled {completed}/{len(episodes)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": record_id, "error": repr(exc)})
                print(f"ERROR {record_id}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete label replay queue")
    summary = {"schema": "policy_geodesic_collection_v1",
               "manifest_sha256": digest(args.manifest),
               "requested_trajectories": len(episodes),
               "completed_trajectories": completed, "errors": errors}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or completed != len(episodes):
        raise RuntimeError(f"geodesic label coverage {completed}/{len(episodes)}")


if __name__ == "__main__":
    main()
