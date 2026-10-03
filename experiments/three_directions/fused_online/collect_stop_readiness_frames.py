"""Replay frozen train-only STOP probe trajectories and save endpoint RGB.

Replay order is scene-local to avoid repeatedly reloading MP3D meshes. The
manifest order and episode set remain fixed for downstream analysis.
"""

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


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_scene(scene_id):
    prefix = "data/scene_datasets/"
    return scene_id[len(prefix):] if scene_id.startswith(prefix) else scene_id


def action_steps(action):
    if action.startswith("move forward ") and action.endswith("cm"):
        distance = int(action.split()[-1][:-2])
        if distance in (25, 50, 75):
            return 1, distance // 25
    if action.startswith("turn left ") or action.startswith("turn right "):
        words = action.split()
        angle = int(words[2])
        if angle in (15, 30, 45) and words[3] == "degrees":
            return (2 if words[1] == "left" else 3), angle // 15
    raise ValueError(f"unknown action: {action}")


def save_frame(observation, path):
    image = Image.fromarray(observation["rgb"]).convert("RGB")
    image.thumbnail((336, 336))
    image.save(path, quality=82, optimize=True)


def collect_one(env, observation, plan, output):
    record_id = plan["record_id"]
    episode = env.current_episode
    assert str(episode.episode_id) == plan["episode_id"]
    assert canonical_scene(str(episode.scene_id)) == plan["scene_id"]
    assert observation["instruction"]["text"].strip() == plan["instruction"].strip()
    actions = plan["actions"]
    assert actions[-1] == "stop" and "stop" not in actions[:-1]
    folder = output / "frames" / record_id
    folder.mkdir(parents=True, exist_ok=True)
    first = folder / "initial.jpg"
    last = folder / "terminal.jpg"
    save_frame(observation, first)
    for action in actions[:-1]:
        code, repeat = action_steps(action)
        for _ in range(repeat):
            if env.episode_over:
                raise RuntimeError(f"replay ended early: {record_id}")
            observation = env.step({"action": code})
    save_frame(observation, last)
    distance = float(env.get_metrics()["distance_to_goal"])
    expected = plan["terminal_distance_m_for_replay_audit_only"]
    if abs(distance - expected) > 0.25:
        raise RuntimeError(f"trajectory drift {record_id}: {distance:.3f} vs {expected:.3f}")
    return {
        "record_id": record_id,
        "episode_id": plan["episode_id"],
        "scene_id": plan["scene_id"],
        "instruction": plan["instruction"],
        "task_success": plan["task_success"],
        "end_reason": plan["end_reason"],
        "motion_actions": len(actions) - 1,
        "frames": [
            {"action_index": 0, "image": str(first.relative_to(output))},
            {"action_index": len(actions) - 1, "image": str(last.relative_to(output))},
        ],
        "replayed_terminal_distance_m_for_audit_only": distance,
        "source_terminal_distance_m_for_audit_only": expected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["schema"] == "stop_readiness_onpolicy_train_v1"
    assert digest(DATASET) == manifest["train_dataset_sha256"]
    plans = sorted(manifest["records"], key=lambda row: (
        row["scene_id"], row["episode_id"], row["record_id"]))
    if args.limit_records:
        assert 0 < args.limit_records <= len(plans)
        plans = plans[:args.limit_records]
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
    assert len(by_id) == len(dataset.episodes)
    requested = defaultdict(deque)
    selected = []
    for row in plans:
        episode = by_id[row["episode_id"]]
        assert canonical_scene(str(episode.scene_id)) == row["scene_id"]
        selected.append(episode)
        requested[row["episode_id"]].append(row)
    dataset.episodes = selected
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected replay episode {eid}")
            plan = requested[eid].popleft()
            path = output / "records" / f"{plan['record_id']}.json"
            path.parent.mkdir(exist_ok=True)
            if path.is_file():
                previous = json.loads(path.read_text())
                if previous["record_id"] == plan["record_id"] and all(
                    (output / frame["image"]).is_file() for frame in previous["frames"]
                ):
                    processed += 1
                    continue
            try:
                record = collect_one(env, observation, plan, output)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, path)
                processed += 1
                if processed % 20 == 0:
                    print(f"replayed {processed}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": plan["record_id"], "error": repr(exc)})
                print(f"ERROR {plan['record_id']}: {exc!r}", flush=True)
    assert not any(requested.values())
    summary = {
        "schema": "stop_readiness_frame_collection_v1",
        "manifest_sha256": digest(args.manifest),
        "requested_trajectories": len(selected),
        "completed_trajectories": processed,
        "errors": errors,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"requested": len(selected), "completed": processed,
                      "errors": len(errors)}), flush=True)
    if errors or processed != len(selected):
        raise RuntimeError("replay collection incomplete")


if __name__ == "__main__":
    main()
