"""Replay train-only failed pairs with the online reward's turn sampling.

All four scored views come from the initial image and turn boundaries, just
as in the online fused-reward hook. Terminal simulator distance is checked
against the source rollout and is never stored as model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import habitat
from habitat import Env
from PIL import Image


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


def action_steps(action: str) -> tuple[int, int]:
    words = action.split()
    if action.startswith("move forward ") and words[-1].endswith("cm"):
        distance = int(words[-1][:-2])
        if distance in (25, 50, 75):
            return 1, distance // 25
    if action.startswith("turn left ") or action.startswith("turn right "):
        angle = int(words[2])
        if angle in (15, 30, 45) and words[3] == "degrees":
            return (2 if words[1] == "left" else 3), angle // 15
    raise ValueError(f"unknown R2R motion action: {action}")


def collect_one(env: Env, observation: dict, plan: dict,
                output: Path, manifest_sha256: str) -> dict:
    row, role = plan["pair"], plan["role"]
    trajectory = row[role]
    eid = str(row["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != row["scene_id"]:
        raise ValueError("Habitat replay episode/scene mismatch")
    if observation["instruction"]["text"].strip() != row["instruction"].strip():
        raise ValueError("replay instruction mismatch")
    actions = [action for action in trajectory["executed_actions"] if action != "stop"]
    boundaries = trajectory["turn_action_boundaries"]
    if len(actions) < 3 or not boundaries or boundaries[-1] != len(actions) or \
            any(a > b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("invalid action/turn boundaries")
    images = [Image.fromarray(observation["rgb"]).convert("RGB")]
    action_index = 0
    for boundary in boundaries:
        while action_index < boundary:
            if env.episode_over:
                raise RuntimeError("replay ended before next turn")
            code, repeat = action_steps(actions[action_index])
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError("replay ended within composite action")
                observation = env.step({"action": code})
            action_index += 1
        images.append(Image.fromarray(observation["rgb"]).convert("RGB"))
    distance = float(env.get_metrics()["distance_to_goal"])
    recorded = float(trajectory["terminal_distance_m_for_replay_audit_only"])
    if abs(distance - recorded) > 0.25:
        raise RuntimeError(f"trajectory drift: replay={distance:.3f} log={recorded:.3f}")
    if trajectory["task_success"] or distance < 3.5:
        raise ValueError("invalid failed-trajectory label")
    indices = [round((len(images) - 1) * index / 3) for index in range(4)]
    record_id = row["pair_id"] + "_" + role
    folder = output / "frames" / record_id
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for ordinal, index in enumerate(indices):
        image = images[index]
        image.thumbnail((336, 336))
        path = folder / f"{ordinal:04d}.jpg"
        image.save(path, quality=82, optimize=True)
        frames.append({"turn_index": index, "initial": index == 0,
                       "image": str(path.relative_to(output))})
    return {"record_id": record_id, "pair_id": row["pair_id"], "role": role,
            "manifest_sha256": manifest_sha256, "episode_id": row["episode_id"],
            "scene_id": row["scene_id"], "split": row["split"],
            "instruction": row["instruction"], "motion_actions": len(actions),
            "turns": len(boundaries), "frames": frames,
            "replayed_terminal_distance_m_for_audit_only": distance,
            "source_terminal_distance_m_for_audit_only": recorded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "failure_rank_train_scene_v1" or \
            digest(DATASET) != manifest["train_sha256"]:
        raise ValueError("manifest or train source mismatch")
    manifest_sha256 = digest(args.manifest)
    if args.limit_pairs < 0 or args.limit_pairs > len(manifest["pairs"]):
        raise ValueError("invalid pair limit")
    pairs = manifest["pairs"][:args.limit_pairs or None]
    plans = [{"pair": row, "role": role} for row in pairs
             for role in ("near", "far")]
    allowed = set().union(*map(set, manifest["scene_split"].values()))
    if {row["scene_id"] for row in pairs} - allowed:
        raise ValueError("scene split mismatch")
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
    by_id = {str(episode.episode_id): episode for episode in dataset.episodes}
    if len(by_id) != len(dataset.episodes):
        raise ValueError("ambiguous train episode IDs")
    plans.sort(key=lambda item: (item["pair"]["scene_id"], item["pair"]["episode_id"],
                                 item["pair"]["pair_id"], item["role"]))
    requested = defaultdict(deque)
    selected = []
    for plan in plans:
        eid = str(plan["pair"]["episode_id"])
        episode = by_id.get(eid)
        if episode is None or canonical_scene(str(episode.scene_id)) != plan["pair"]["scene_id"]:
            raise ValueError(f"manifest episode missing or wrong scene {eid}")
        selected.append(episode)
        requested[eid].append(plan)
    dataset.episodes = selected
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    processed, errors = 0, []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected replay episode {eid}")
            plan = requested[eid].popleft()
            record_id = plan["pair"]["pair_id"] + "_" + plan["role"]
            record_path = output / "records" / f"{record_id}.json"
            record_path.parent.mkdir(exist_ok=True)
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                if record["record_id"] == record_id and \
                        record["manifest_sha256"] == manifest_sha256 and \
                        len(record["frames"]) == 4 and all(
                            (output / frame["image"]).is_file() for frame in record["frames"]):
                    processed += 1
                    continue
            try:
                record = collect_one(env, observation, plan, output, manifest_sha256)
                temp = record_path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, record_path)
                processed += 1
                if processed % 20 == 0:
                    print(f"replayed {processed}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": record_id, "error": repr(exc)})
                print(f"ERROR {record_id}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("Habitat replay coverage incomplete")
    summary = {"pairs": len(pairs), "requested_trajectories": len(selected),
               "completed_trajectories": processed, "errors": errors,
               "manifest_sha256": manifest_sha256,
               "sampling": "four evenly spaced initial/turn-boundary views"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or processed != len(selected):
        raise RuntimeError(f"replay coverage {processed}/{len(selected)}")


if __name__ == "__main__":
    main()
