"""Render sparse R2R-train expert frames for ordinal progress learning.

Run from the ActiveVLN repository root with its Habitat server environment.
Images and per-episode records remain on the experiment host, outside Git.
The script is resumable and never reads val-unseen scenes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import habitat
from habitat import Env
from PIL import Image

from ordinal_progress_selection import select_episode_ids


CONFIG = "vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"
DATASET = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
GT = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz")


def canonical_scene(scene_id: str) -> str:
    prefix = "data/scene_datasets/"
    return scene_id[len(prefix):] if scene_id.startswith(prefix) else scene_id


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_at(observation, path: Path):
    frame = Image.fromarray(observation["rgb"]).convert("RGB")
    frame.thumbnail((336, 336))
    frame.save(path, quality=82, optimize=True)


def collect_one(env, observation, gt, output: Path):
    episode = env.current_episode
    episode_id = str(episode.episode_id)
    actions = list(gt[episode_id]["actions"])
    if len(actions) < 2 or actions[-1] != 0 or any(a not in (0, 1, 2, 3) for a in actions):
        raise ValueError(f"invalid expert actions for {episode_id}")
    if 0 in actions[:-1]:
        raise ValueError(f"early STOP in expert path {episode_id}")
    motion_count = len(actions) - 1
    # At most six images per episode. Preserve temporal order in the manifest.
    points = sorted({round(motion_count * j / 5) for j in range(6)})
    folder = output / "frames" / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    records = []

    def save(point: int):
        image_path = folder / f"{point:04d}.jpg"
        image_at(observation, image_path)
        records.append({"action_index": point,
                        "progress_fraction": point / motion_count,
                        "image": str(image_path.relative_to(output))})

    if 0 in points:
        save(0)
    for index, action in enumerate(actions[:-1], 1):
        if env.episode_over:
            raise RuntimeError(f"episode {episode_id} ended at action {index}/{motion_count}")
        observation = env.step({"action": int(action)})
        if index in points:
            save(index)
    if not records or records[-1]["action_index"] != motion_count:
        raise RuntimeError(f"final expert frame missing for {episode_id}")
    return {"episode_id": int(episode_id), "scene_id": canonical_scene(str(episode.scene_id)),
            "instruction": observation["instruction"]["text"],
            "motion_actions": motion_count, "frames": records,
            "final_distance_to_goal_for_audit_only": float(env.get_metrics()["distance_to_goal"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional first N episodes for a smoke run")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if digest(DATASET) != manifest["sources"]["train_sha256"] or \
            digest(GT) != manifest["sources"]["ground_truth_sha256"]:
        raise ValueError("source dataset checksum mismatch")
    requested_ids = select_episode_ids(manifest, args.subset, args.limit)
    requested = set(map(str, requested_ids))
    allowed_scenes = set(manifest[f"{args.subset}_scenes"])
    output = args.output / args.subset
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(GT, "rt", encoding="utf-8") as stream:
        ground_truth = json.load(stream)

    from VLN_CE.vlnce_baselines.config.default import get_config

    config = get_config(CONFIG)
    config.defrost()
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "train"
    config.TASK_CONFIG.TASK.NDTW.SPLIT = "train"
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.TASK_CONFIG.freeze()
    config.freeze()
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    selected = [episode for episode in dataset.episodes
                if str(episode.episode_id) in requested]
    selected.sort(key=lambda episode: int(episode.episode_id))
    if len(selected) != len(requested):
        raise ValueError(f"selected {len(selected)} of {len(requested)} requested IDs")
    if {canonical_scene(str(episode.scene_id)) for episode in selected} - allowed_scenes:
        raise ValueError("scene split violation")
    dataset.episodes = selected
    selected_by_id = {str(episode.episode_id): episode for episode in selected}
    summary = {"subset": args.subset, "requested": len(selected),
               "manifest_sha256": digest(args.manifest),
               "episodes": [], "errors": []}
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        seen = set()
        for _ in selected:
            # Habitat's first reset can advance past dataset.episodes[0].
            # Always key saved data by the actual reset episode.
            observation = env.reset()
            episode = env.current_episode
            episode_id = str(episode.episode_id)
            if episode_id not in selected_by_id or episode_id in seen:
                raise RuntimeError(f"unexpected or duplicate Habitat episode {episode_id}")
            seen.add(episode_id)
            record_file = output / "records" / f"{episode_id}.json"
            record_file.parent.mkdir(exist_ok=True)
            if record_file.exists():
                record = json.loads(record_file.read_text())
                if record["episode_id"] == int(episode_id) and all(
                    (output / frame["image"]).is_file() for frame in record["frames"]
                ):
                    summary["episodes"].append(episode_id)
                    continue
            try:
                record = collect_one(env, observation, ground_truth, output)
                if record["episode_id"] != int(episode_id):
                    raise ValueError("record ID differs from Habitat episode")
                if record["scene_id"] != canonical_scene(str(episode.scene_id)):
                    raise ValueError("Habitat episode order mismatch")
                temp = record_file.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, record_file)
                summary["episodes"].append(episode_id)
                print(f"{args.subset} {len(summary['episodes'])}/{len(selected)} episode={episode_id}", flush=True)
            except Exception as exc:
                summary["errors"].append({"episode_id": episode_id, "error": repr(exc)})
                print(f"ERROR episode={episode_id}: {exc!r}", flush=True)
        if seen != requested:
            raise RuntimeError(f"Habitat episode coverage: {len(seen)}/{len(requested)}")
    summary["completed"] = len(summary["episodes"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["errors"] or summary["completed"] != summary["requested"]:
        raise RuntimeError(f"incomplete collection: {summary['completed']}/{summary['requested']}")


if __name__ == "__main__":
    main()
