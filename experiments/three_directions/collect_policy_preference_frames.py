"""Replay selected train-only policy outcomes and cache sparse RGB histories.

The manifest fixes one success and one unambiguous failure from each group
of four. Only executed actions are replayed; a model's STOP token is not a
simulator step in the original server. A terminal distance check rejects
trajectory drift before any visual preference fitting.
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


def action_steps(action: str) -> tuple[int | None, int]:
    if action == "stop":
        return None, 0
    words = action.split()
    if action.startswith("move forward ") and words[-1].endswith("cm"):
        distance = int(words[-1][:-2])
        if distance in (25, 50, 75):
            return 1, distance // 25
    if action.startswith("turn left ") or action.startswith("turn right "):
        angle = int(words[2])
        if angle in (15, 30, 45) and words[3] == "degrees":
            return (2 if words[1] == "left" else 3), angle // 15
    raise ValueError(f"unknown R2R action: {action}")


def save_frame(observation: dict, path: Path) -> None:
    frame = Image.fromarray(observation["rgb"]).convert("RGB")
    frame.thumbnail((336, 336))
    frame.save(path, quality=82, optimize=True)


def collect_one(env: Env, observation: dict, plan: dict, output: Path) -> dict:
    row, role = plan["pair"], plan["role"]
    trajectory = row[role]
    eid = str(row["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != row["scene_id"]:
        raise ValueError("Habitat replay episode/scene mismatch")
    if observation["instruction"]["text"].strip() != row["instruction"].strip():
        raise ValueError("replay instruction mismatch")
    actions = trajectory["executed_actions"]
    motion = [action for action in actions if action != "stop"]
    if len(motion) < 3 or "stop" in actions[:-1]:
        raise ValueError("too few policy actions or early stop")
    points = sorted({round(len(motion) * index / 3) for index in range(4)})
    record_id = row["pair_id"] + "_" + role
    folder = output / "frames" / record_id
    folder.mkdir(parents=True, exist_ok=True)
    frames = []

    def save(index: int) -> None:
        file = folder / f"{index:04d}.jpg"
        save_frame(observation, file)
        frames.append({"action_index": index, "image": str(file.relative_to(output))})

    save(0)
    for index, action in enumerate(motion, 1):
        code, repeat = action_steps(action)
        for _ in range(repeat):
            if env.episode_over:
                raise RuntimeError(f"replay ended early {record_id} action={index}")
            observation = env.step({"action": code})
        if index in points:
            save(index)
    distance = float(env.get_metrics()["distance_to_goal"])
    recorded = trajectory["terminal_distance_m_for_replay_audit_only"]
    if abs(distance - recorded) > 0.25:
        raise RuntimeError(f"trajectory drift {record_id}: replay={distance:.3f} log={recorded:.3f}")
    if (role == "success") != bool(trajectory["task_success"]):
        raise ValueError("preference label mismatch")
    if role == "success" and distance > 3.0 or role == "failure" and distance < 3.5:
        raise RuntimeError("replay outcome threshold mismatch")
    if frames[-1]["action_index"] != len(motion):
        raise RuntimeError("terminal RGB missing")
    return {"record_id": record_id, "pair_id": row["pair_id"], "role": role,
            "episode_id": row["episode_id"], "scene_id": row["scene_id"],
            "split": row["split"], "instruction": row["instruction"],
            "motion_actions": len(motion), "frames": frames,
            "replayed_terminal_distance_m_for_audit_only": distance,
            "source_terminal_distance_m_for_audit_only": recorded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if digest(DATASET) != manifest["train_sha256"]:
        raise ValueError("train source checksum mismatch")
    pairs = manifest["pairs"][:args.limit_pairs or None]
    if not pairs or args.limit_pairs < 0 or \
            len(pairs) != (args.limit_pairs or len(manifest["pairs"])):
        raise ValueError("invalid pair limit")
    plans = [{"pair": row, "role": role} for row in pairs
             for role in ("success", "failure")]
    desired_scenes = {row["scene_id"] for row in pairs}
    allowed = set().union(*map(set, manifest["scene_split"].values()))
    if desired_scenes - allowed:
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
                raise RuntimeError(f"unexpected or duplicate replay episode {eid}")
            plan = requested[eid].popleft()
            row, role = plan["pair"], plan["role"]
            record_id = row["pair_id"] + "_" + role
            record_path = output / "records" / f"{record_id}.json"
            record_path.parent.mkdir(exist_ok=True)
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                if record["record_id"] == record_id and len(record["frames"]) >= 3 and all(
                    (output / frame["image"]).is_file() for frame in record["frames"]
                ):
                    processed += 1
                    continue
            try:
                record = collect_one(env, observation, plan, output)
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
               "manifest_sha256": digest(args.manifest)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or processed != len(selected):
        raise RuntimeError(f"replay coverage {processed}/{len(selected)}")


if __name__ == "__main__":
    main()
