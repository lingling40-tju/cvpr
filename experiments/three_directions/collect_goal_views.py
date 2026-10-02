"""Cache four RGB views at R2R-train goal poses without expert-path replay.

This is a representation-data diagnostic. Goal coordinates are used only to
prepare train-split positives, never as an inference input or RL reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import habitat
import numpy as np
from habitat import Env
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_coeffs
from PIL import Image

from ordinal_progress_selection import select_episode_ids


CONFIG = "vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"
DATASET = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_scene(scene_id: str) -> str:
    prefix = "data/scene_datasets/"
    return scene_id[len(prefix):] if scene_id.startswith(prefix) else scene_id


def collect_one(env: Env, observation: dict, output: Path) -> dict:
    episode = env.current_episode
    eid = str(episode.episode_id)
    goal = list(episode.goals[0].position)
    initial = quat_from_coeffs(episode.start_rotation)
    views = []
    folder = output / "views" / eid
    folder.mkdir(parents=True, exist_ok=True)
    for quarter in range(4):
        rotation = quat_to_coeffs(quat_from_angle_axis(quarter * np.pi / 2,
                                                      np.array([0, 1, 0])) * initial)
        rgb = env.sim.get_observations_at(goal, rotation, keep_agent_at_new_pose=False)
        if rgb is None or "rgb" not in rgb:
            raise RuntimeError(f"goal view unavailable episode={eid} quarter={quarter}")
        file = folder / f"q{quarter}.jpg"
        frame = Image.fromarray(rgb["rgb"]).convert("RGB")
        frame.thumbnail((336, 336))
        frame.save(file, quality=82, optimize=True)
        views.append({"quarter": quarter, "image": str(file.relative_to(output))})
    return {"episode_id": int(eid), "scene_id": canonical_scene(str(episode.scene_id)),
            "instruction": observation["instruction"]["text"],
            "goal_position_for_data_preparation_only": goal,
            "views": views}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if digest(DATASET) != manifest["sources"]["train_sha256"]:
        raise ValueError("train dataset checksum mismatch")
    requested_ids = select_episode_ids(manifest, args.subset, args.limit)
    requested = set(map(str, requested_ids))
    allowed_scenes = set(manifest[f"{args.subset}_scenes"])
    output = args.output / args.subset
    output.mkdir(parents=True, exist_ok=True)
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
    selected = [episode for episode in dataset.episodes
                if str(episode.episode_id) in requested]
    # Habitat reloads a scene on a scene switch. Grouping by scene makes the
    # larger screen substantially cheaper without changing selected IDs.
    selected.sort(key=lambda episode: (canonical_scene(str(episode.scene_id)),
                                       int(episode.episode_id)))
    if len(selected) != len(requested):
        raise ValueError("manifest episode coverage mismatch")
    if {canonical_scene(str(e.scene_id)) for e in selected} - allowed_scenes:
        raise ValueError("manifest scene split violation")
    dataset.episodes = selected
    seen, errors = set(), []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if eid not in requested or eid in seen:
                raise RuntimeError(f"unexpected or duplicate episode {eid}")
            seen.add(eid)
            record_path = output / "records" / f"{eid}.json"
            record_path.parent.mkdir(exist_ok=True)
            if record_path.is_file():
                old = json.loads(record_path.read_text())
                if old["episode_id"] == int(eid) and len(old["views"]) == 4 and all(
                    (output / view["image"]).is_file() for view in old["views"]
                ):
                    print(f"cached {len(seen)}/{len(selected)} episode={eid}", flush=True)
                    continue
            try:
                record = collect_one(env, observation, output)
                if record["scene_id"] != canonical_scene(str(env.current_episode.scene_id)):
                    raise ValueError("scene mismatch")
                temp = record_path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, record_path)
                print(f"rendered {len(seen)}/{len(selected)} episode={eid}", flush=True)
            except Exception as exc:
                errors.append({"episode_id": eid, "error": repr(exc)})
                print(f"ERROR episode={eid} {exc!r}", flush=True)
    if seen != requested:
        raise RuntimeError("episode enumeration coverage mismatch")
    summary = {"subset": args.subset, "requested": len(requested),
               "completed": len(requested) - len(errors), "errors": errors,
               "manifest_sha256": digest(args.manifest)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"goal view coverage {summary['completed']}/{len(requested)}")


if __name__ == "__main__":
    main()
