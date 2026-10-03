"""Cache one verified initial RGB image per group-four preference episode.

Existing policy-history replays are reused only when their manifest, scene,
instruction and image agree. Habitat renders only episodes absent from that
cache. Model inputs receive RGB and instruction, never terminal labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from collect_policy_preference_frames import CONFIG, DATASET, canonical_scene, digest, save_frame


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    old_manifest = json.loads(args.old_manifest.read_text())
    if manifest["schema"] != "group4_first_action_preference_manifest_v1" or \
            old_manifest["schema"] != "policy_process_train_manifest_v1" or \
            manifest["train_dataset_sha256"] != digest(DATASET) or \
            old_manifest["train_dataset_sha256"] != manifest["train_dataset_sha256"]:
        raise ValueError("manifest or train source mismatch")
    expected_sources = {key: value["sha256"] for key, value in manifest["sources"].items()}
    if expected_sources != {key: value["sha256"] for key, value in old_manifest["sources"].items()}:
        raise ValueError("rollout source mismatch")
    part = args.part
    rows = manifest["selected"][part]
    by_id = defaultdict(list)
    for row in rows:
        by_id[str(row["episode_id"])].append(row)
    for eid, episode_rows in by_id.items():
        if len({(row["scene_id"], row["instruction"]) for row in episode_rows}) != 1:
            raise ValueError(f"inconsistent episode identity {eid}")
    old_part = args.old_root / part
    old_sha = digest(args.old_manifest)
    candidates = defaultdict(list)
    for path in (old_part / "records").glob("*.json"):
        record = json.loads(path.read_text())
        eid = str(record["episode_id"])
        if eid in by_id:
            candidates[eid].append(record)
    output = args.output_root / part
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    index = {}
    missing = []
    for eid in sorted(by_id):
        row = by_id[eid][0]
        records = candidates[eid]
        hashes = set()
        source_image = None
        for record in records:
            path = old_part / record["initial_image"]
            if record["manifest_sha256"] != old_sha or \
                    record["scene_id"] != row["scene_id"] or \
                    record["instruction"].strip() != row["instruction"].strip() or \
                    not path.is_file():
                raise ValueError(f"invalid reused initial observation {part}/{eid}")
            hashes.add(digest(path))
            source_image = path
        if len(hashes) > 1:
            raise ValueError(f"initial image differs by rollout {part}/{eid}")
        target = frames / f"{eid}.jpg"
        if source_image is None:
            missing.append(eid)
            continue
        if target.exists():
            if digest(target) != next(iter(hashes)):
                raise ValueError(f"existing image mismatch {part}/{eid}")
        else:
            os.link(source_image, target)
        index[eid] = {"scene_id": row["scene_id"],
                      "instruction": row["instruction"],
                      "image": str(target.relative_to(output)),
                      "image_sha256": next(iter(hashes)),
                      "source": "verified_policy_replay"}
    if missing:
        import habitat
        from habitat import Env
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
        all_episodes = {str(episode.episode_id): episode for episode in dataset.episodes}
        if len(all_episodes) != len(dataset.episodes):
            raise ValueError("ambiguous Habitat episode IDs")
        selected = []
        for eid in sorted(missing, key=lambda key: (by_id[key][0]["scene_id"], key)):
            episode = all_episodes.get(eid)
            if episode is None or canonical_scene(str(episode.scene_id)) != by_id[eid][0]["scene_id"]:
                raise ValueError(f"missing Habitat episode {eid}")
            selected.append(episode)
        dataset.episodes = selected
        seen = set()
        with Env(config.TASK_CONFIG, dataset=dataset) as env:
            for _ in selected:
                observation = env.reset()
                eid = str(env.current_episode.episode_id)
                if eid not in missing or eid in seen:
                    raise ValueError(f"unexpected or repeated Habitat episode {eid}")
                seen.add(eid)
                row = by_id[eid][0]
                actual_scene = canonical_scene(str(env.current_episode.scene_id))
                actual_instruction = observation["instruction"]["text"].strip()
                if actual_scene != row["scene_id"] or \
                        actual_instruction != row["instruction"].strip():
                    raise ValueError(f"initial Habitat source mismatch {eid}: "
                                     f"scene={actual_scene == row['scene_id']}, "
                                     f"instruction={actual_instruction == row['instruction'].strip()}")
                target = frames / f"{eid}.jpg"
                if not target.exists():
                    save_frame(observation, target)
                index[eid] = {"scene_id": row["scene_id"],
                              "instruction": row["instruction"],
                              "image": str(target.relative_to(output)),
                              "image_sha256": digest(target),
                              "source": "new_initial_habitat_render"}
        if seen != set(missing):
            raise ValueError("Habitat reset did not cover every missing episode")
    if set(index) != set(by_id):
        raise ValueError("incomplete initial image coverage")
    result = {"schema": "group4_first_action_image_index_v1",
              "part": part, "manifest_sha256": digest(args.manifest),
              "old_manifest_sha256": old_sha,
              "groups": len(rows), "unique_episodes": len(index),
              "reused_episodes": len(index) - len(missing),
              "rendered_episodes": len(missing),
              "images": index}
    atomic_json(output / "index.json", result)
    print(json.dumps({key: result[key] for key in
                      ("part", "groups", "unique_episodes", "reused_episodes",
                       "rendered_episodes", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
