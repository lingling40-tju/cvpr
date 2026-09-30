"""Collect val-unseen RGB transitions without querying the event verifier.

The output is an annotation bundle. Human event labels must be added before
calling the frozen verifier; this script records no completion verdicts.
Run with ActiveVLN's Habitat environment from its repository root.
"""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import habitat
import numpy as np
from habitat import Env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from PIL import Image


CONFIG = "vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"
ACTION_TEXT = {0: "stop", 1: "move forward 25cm", 2: "turn left 15 degrees", 3: "turn right 15 degrees"}


def parse_events(instruction):
    request = Request(
        "http://127.0.0.1:5003/parse",
        json.dumps({"instruction": instruction}).encode(),
        {"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result.get("model") == "Qwen3-VL-8B-Instruct", result
    return result["events"]


def choose_episodes(episodes, excluded_ids, count):
    groups = {}
    for episode in episodes:
        if str(episode.episode_id) not in excluded_ids:
            groups.setdefault(str(episode.scene_id), []).append(episode)
    selected = []
    for round_index in range(max(map(len, groups.values()))):
        for scene in sorted(groups):
            if round_index < len(groups[scene]):
                selected.append(groups[scene][round_index])
                if len(selected) == count:
                    return selected
    return selected


def save_rgb(observation, path):
    Image.fromarray(observation["rgb"]).save(path, quality=85)


def collect_episode(env, output):
    observation = env.reset()
    episode = env.current_episode
    episode_id = str(episode.episode_id)
    instruction = observation["instruction"]["text"]
    events = parse_events(instruction)
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)
    folder = output / "frames" / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    turn = 0
    goal = episode.goals[0].position
    while not env.episode_over and turn < 40:
        before_path = folder / f"{turn:02d}_before.jpg"
        save_rgb(observation, before_path)
        before_position = np.asarray(env.sim.get_agent_state().position, dtype=float)
        actions = []
        for _ in range(3):
            action = follower.get_next_action(goal)
            action = 0 if action is None else int(action)
            actions.append(ACTION_TEXT[action])
            observation = env.step({"action": action})
            if env.episode_over or action == 0:
                break
        after_path = folder / f"{turn:02d}_after.jpg"
        save_rgb(observation, after_path)
        delta = np.asarray(env.sim.get_agent_state().position, dtype=float) - before_position
        records.append({
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "instruction": instruction,
            "turn": turn,
            "actions": actions,
            "displacement": float(np.linalg.norm(delta)),
            "vertical_delta": float(delta[1]),
            "before": str(before_path.relative_to(output)),
            "after": str(after_path.relative_to(output)),
            "distance_to_goal": float(env.get_metrics()["distance_to_goal"]),
            "events": events,
        })
        turn += 1
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-manifest", required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    excluded = set(json.loads(Path(args.exclude_manifest).read_text())["episode_ids"])

    from VLN_CE.vlnce_baselines.config.default import get_config

    config = get_config(CONFIG)
    assert config.TASK_CONFIG.DATASET.SPLIT == "val_unseen"
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes = choose_episodes(dataset.episodes, excluded, args.count)
    assert len(dataset.episodes) == args.count
    all_records = []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in dataset.episodes:
            records = collect_episode(env, output)
            all_records.extend(records)
            print(records[0]["episode_id"], len(records), len(records[0]["events"]), flush=True)
    (output / "transitions.json").write_text(json.dumps(all_records, indent=2) + "\n")
    print(f"wrote {len(all_records)} turns across {args.count} episodes", flush=True)


if __name__ == "__main__":
    main()
