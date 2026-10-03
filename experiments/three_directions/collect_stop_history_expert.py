"""Replay frozen R2R-train expert routes into normal egocentric RGB histories.

The records contain executed action text and per-turn RGB. Simulator
distances are stored for labeling/audit only, never for model inputs.
Each route is collected once, resumably; fit/development/audit use
scene-disjoint output directories and can run on separate Habitat GPUs.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import json
import math
import os
from pathlib import Path

import habitat
from habitat import Env

from collect_policy_preference_frames import CONFIG, DATASET, canonical_scene, digest, save_frame


GT = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz")


def macros(actions: list[int]) -> list[tuple[str, list[int]]]:
    if not actions or any(action not in (1, 2, 3) for action in actions):
        raise ValueError("invalid expert motion actions")
    result = []
    index = 0
    while index < len(actions):
        code = actions[index]
        end = index + 1
        while end < len(actions) and actions[end] == code and end - index < 3:
            end += 1
        length = end - index
        if code == 1:
            text = f"move forward {length * 25}cm"
        else:
            text = f"turn {'left' if code == 2 else 'right'} {length * 15} degrees"
        result.append((text, actions[index:end]))
        index = end
    if [code for _, group in result for code in group] != actions:
        raise ValueError("macro compression changed path")
    return result


def collect_one(env: Env, observation: dict, plan: dict, gt: dict,
                output: Path, manifest_sha: str) -> dict:
    episode_id = plan["episode_id"]
    if str(env.current_episode.episode_id) != episode_id or \
            canonical_scene(str(env.current_episode.scene_id)) != plan["scene_id"]:
        raise ValueError("Habitat episode or scene mismatch")
    with_instruction = observation["instruction"]["text"].strip()
    if with_instruction != plan["instruction"].strip() or \
            with_instruction == plan["wrong_instruction"].strip():
        raise ValueError("instruction mismatch or invalid natural swap")
    actions = list(gt[episode_id]["actions"])
    if actions[-1] != 0 or 0 in actions[:-1]:
        raise ValueError("invalid expert STOP placement")
    grouped = macros(actions[:-1])
    folder = output / "frames" / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    initial = folder / "0000.jpg"
    save_frame(observation, initial)
    initial_distance = float(env.get_metrics()["distance_to_goal"])
    if not math.isfinite(initial_distance) or initial_distance < 3.5:
        raise ValueError(f"invalid far-STOP label at start: {initial_distance}")
    turns = []
    action_count = 0
    for offset in range(0, len(grouped), 3):
        action_group = grouped[offset:offset + 3]
        for _, atomic in action_group:
            for code in atomic:
                if env.episode_over:
                    raise RuntimeError("expert route ended before STOP")
                observation = env.step({"action": code})
                action_count += 1
        image_path = folder / f"{len(turns) + 1:04d}.jpg"
        save_frame(observation, image_path)
        turns.append({
            "assistant_response": ", ".join(text for text, _ in action_group),
            "image": str(image_path.relative_to(output)),
            "atomic_actions": action_count,
            "distance_to_goal_for_label_only": float(env.get_metrics()["distance_to_goal"]),
        })
    final_distance = float(env.get_metrics()["distance_to_goal"])
    if action_count != plan["motion_actions"] or not math.isfinite(final_distance):
        raise ValueError("expert action count or final metric mismatch")
    return {
        "schema": "stop_history_expert_record_v1",
        "manifest_sha256": manifest_sha,
        "episode_id": episode_id,
        "trajectory_id": plan["trajectory_id"],
        "scene_id": plan["scene_id"],
        "instruction": with_instruction,
        "wrong_instruction": plan["wrong_instruction"],
        "wrong_instruction_episode_id": plan["swap_episode_id"],
        "initial_image": str(initial.relative_to(output)),
        "turns": turns,
        "start_distance_to_goal_for_label_only": initial_distance,
        "end_distance_to_goal_for_label_only": final_distance,
        "end_within_3m": final_distance <= 3.0,
        "motion_actions": action_count,
        "macro_actions": len(grouped),
        "turn_count": len(turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "stop_history_expert_manifest_v1" or \
            digest(DATASET) != manifest["source_sha256"]["train_dataset"] or \
            digest(GT) != manifest["source_sha256"]["ground_truth"]:
        raise ValueError("manifest or source checksum mismatch")
    plans = manifest["selected"][args.part]
    if args.limit < 0:
        raise ValueError("negative limit")
    plans = plans[:args.limit or None]
    requested = {plan["episode_id"]: plan for plan in plans}
    if len(requested) != len(plans):
        raise ValueError("duplicate selected episode")
    output = args.output_root / args.part
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(GT, "rt", encoding="utf-8") as stream:
        ground_truth = json.load(stream)
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
        raise ValueError("duplicate Habitat episode")
    selected = []
    for plan in plans:
        episode = by_id.get(plan["episode_id"])
        if episode is None or canonical_scene(str(episode.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing Habitat episode {plan['episode_id']}")
        selected.append(episode)
    selected.sort(key=lambda episode: (canonical_scene(str(episode.scene_id)),
                                       str(episode.episode_id)))
    dataset.episodes = selected
    queue = {eid: deque([plan]) for eid, plan in requested.items()}
    completed, errors, seen = [], [], set()
    manifest_sha = digest(args.manifest)
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            episode_id = str(env.current_episode.episode_id)
            if episode_id not in queue or episode_id in seen:
                raise RuntimeError(f"unexpected or repeated reset {episode_id}")
            seen.add(episode_id)
            plan = queue[episode_id].popleft()
            record_file = output / "records" / f"{episode_id}.json"
            record_file.parent.mkdir(exist_ok=True)
            if record_file.is_file():
                prior = json.loads(record_file.read_text())
                if prior["manifest_sha256"] == manifest_sha and \
                        (output / prior["initial_image"]).is_file() and \
                        all((output / turn["image"]).is_file() for turn in prior["turns"]):
                    completed.append(episode_id)
                    continue
            try:
                result = collect_one(env, observation, plan, ground_truth,
                                     output, manifest_sha)
                temp = record_file.with_suffix(".tmp")
                temp.write_text(json.dumps(result, indent=2) + "\n")
                os.replace(temp, record_file)
                completed.append(episode_id)
                if len(completed) % 20 == 0:
                    print(f"{args.part} {len(completed)}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"episode_id": episode_id, "error": repr(exc)})
                print(f"ERROR {args.part} {episode_id}: {exc!r}", flush=True)
    if seen != set(requested) or any(queue.values()):
        raise RuntimeError("incomplete Habitat reset coverage")
    summary = {
        "schema": "stop_history_collection_v1",
        "part": args.part,
        "manifest_sha256": manifest_sha,
        "requested": len(selected),
        "completed": len(completed),
        "completed_episode_ids": completed,
        "errors": errors,
        "smoke_limit": args.limit,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or len(completed) != len(selected):
        raise RuntimeError(f"incomplete {args.part} collection: {len(completed)}/{len(selected)}")


if __name__ == "__main__":
    main()
