"""Replay selected group-four rollouts at turn boundaries for a frozen SFT screen.

The query transcript contains observations and executed action responses, but
never a terminal STOP response or an outcome label. A mixed motion+STOP final
response is truncated to its executed motion and followed by the terminal RGB;
this is a counterfactual extra query, recorded explicitly in the audit.
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
    CONFIG, DATASET, action_steps, canonical_scene, digest, save_frame,
)


def source_infos(manifest: dict, plans: list[dict]) -> dict[tuple[int, str, str], dict]:
    needed = {(p["trajectory"]["seed"], str(p["pair"]["episode_id"]))
              for p in plans}
    groups = {}
    for seed in sorted({item[0] for item in needed}):
        source = manifest["sources"][str(seed)]
        path = Path(source["rollout_path"])
        if digest(path) != source["rollout_sha256"]:
            raise ValueError(f"rollout checksum mismatch seed={seed}")
        with path.open() as stream:
            for line in stream:
                step = json.loads(line)
                for info in step["info"]:
                    eid = str(info["episode_id"])
                    if (seed, eid) in needed:
                        groups.setdefault((seed, eid), []).append(info)
    result = {}
    for plan in plans:
        row, role, trajectory = plan["pair"], plan["role"], plan["trajectory"]
        seed, eid = trajectory["seed"], str(row["episode_id"])
        group = groups.get((seed, eid))
        if group is None or len(group) != 4:
            raise ValueError(f"missing group-four source seed={seed} episode={eid}")
        info = group[trajectory["variant"]]
        actions = [str(action) for turn in info["gen_traj"]
                   for action in turn.get("executed_actions", [])]
        if actions != trajectory["executed_actions"] or \
                info["instruction"].strip() != row["instruction"].strip():
            raise ValueError(f"rollout/manifest mismatch seed={seed} episode={eid}")
        result[(seed, eid, role)] = info
    return result


def collect_one(env: Env, observation: dict, plan: dict, info: dict,
                output: Path) -> dict:
    row, role, trajectory = plan["pair"], plan["role"], plan["trajectory"]
    eid = str(row["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != row["scene_id"] or \
            observation["instruction"]["text"].strip() != row["instruction"].strip():
        raise ValueError("Habitat replay episode, scene, or instruction mismatch")
    record_id = row["pair_id"] + "_" + role
    folder = output / "frames" / record_id
    folder.mkdir(parents=True, exist_ok=True)
    turns = []
    initial = folder / "0000.jpg"
    save_frame(observation, initial)
    motion_count = 0
    terminal_stop_omitted = False
    for turn in info["gen_traj"]:
        executed = list(turn.get("executed_actions", []))
        if not executed:
            continue
        if "stop" in executed[:-1]:
            raise ValueError("early STOP within turn")
        motion = [a for a in executed if a != "stop"]
        for action in motion:
            code, repeat = action_steps(action)
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError(f"replay ended early {record_id}")
                observation = env.step({"action": code})
            motion_count += 1
        if "stop" in executed:
            terminal_stop_omitted = True
        if motion:
            # Match the trainer's response where it does not expose STOP.
            response = ", ".join(motion) if terminal_stop_omitted else turn["response"]
            if "stop" in response.lower():
                raise ValueError("STOP leaked into history")
            path = folder / f"{len(turns) + 1:04d}.jpg"
            save_frame(observation, path)
            turns.append({"assistant_response": response,
                          "image": str(path.relative_to(output)),
                          "motion_actions": motion})
    expected_motion = [a for a in trajectory["executed_actions"] if a != "stop"]
    if motion_count != len(expected_motion):
        raise ValueError("motion count mismatch")
    distance = float(env.get_metrics()["distance_to_goal"])
    source_distance = trajectory["terminal_distance_m_for_replay_audit_only"]
    if abs(distance - source_distance) > .25:
        raise RuntimeError(f"trajectory drift {record_id}: {distance:.3f} vs {source_distance:.3f}")
    if not turns:
        raise ValueError("empty motion history")
    return {"schema": "full_history_record_v1", "record_id": record_id,
            "pair_id": row["pair_id"], "role": role,
            "split": plan["split"], "scene_id": row["scene_id"],
            "episode_id": row["episode_id"], "instruction": row["instruction"],
            "initial_image": str(initial.relative_to(output)), "turns": turns,
            "terminal_stop_response_omitted": terminal_stop_omitted,
            "counterfactual_extra_query": bool(
                info["gen_traj"] and "stop" in info["gen_traj"][-1].get("executed_actions", [])
                and any(a != "stop" for a in info["gen_traj"][-1]["executed_actions"])),
            "motion_actions": motion_count,
            "replayed_terminal_distance_m_for_audit_only": distance,
            "source_terminal_distance_m_for_audit_only": source_distance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=("development", "audit"))
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    probe = json.loads(args.probe_manifest.read_text())
    if probe["source_manifest_sha256"] != digest(args.manifest) or \
            probe["group_size"] != 4 or digest(DATASET) != manifest["train_sha256"]:
        raise ValueError("manifest/source checksum mismatch")
    rows = {row["pair_id"]: row for row in manifest["pairs"]}
    selected = [item for item in probe["pairs"]
                if args.split is None or item["split"] == args.split]
    selected = selected[:args.limit_pairs or None]
    if not selected or args.limit_pairs < 0:
        raise ValueError("empty or invalid pair selection")
    plans = [{"pair": rows[item["pair_id"]], "split": item["split"],
              "role": role, "trajectory": rows[item["pair_id"]][role]}
             for item in selected for role in ("success", "failure")]
    selected_scenes = {item["pair_id"]: item["scene_id"] for item in selected}
    for plan in plans:
        if plan["pair"]["scene_id"] != selected_scenes[plan["pair"]["pair_id"]]:
            raise ValueError("probe scene mismatch")
    infos = source_infos(manifest, plans)
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
        raise ValueError("ambiguous train episode IDs")
    plans.sort(key=lambda p: (p["pair"]["scene_id"], p["pair"]["episode_id"],
                              p["pair"]["pair_id"], p["role"]))
    requested = defaultdict(deque)
    episodes = []
    for plan in plans:
        eid = str(plan["pair"]["episode_id"])
        episode = by_id.get(eid)
        if episode is None or canonical_scene(str(episode.scene_id)) != plan["pair"]["scene_id"]:
            raise ValueError(f"missing replay episode {eid}")
        episodes.append(episode)
        requested[eid].append(plan)
    dataset.episodes = episodes
    args.output.mkdir(parents=True, exist_ok=True)
    completed, errors = 0, []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in episodes:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            plan = requested[eid].popleft()
            row, role = plan["pair"], plan["role"]
            record_id = row["pair_id"] + "_" + role
            record_path = args.output / "records" / f"{record_id}.json"
            record_path.parent.mkdir(exist_ok=True)
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                if record["schema"] == "full_history_record_v1" and \
                        (args.output / record["initial_image"]).is_file() and \
                        all((args.output / turn["image"]).is_file() for turn in record["turns"]):
                    completed += 1
                    continue
            try:
                key = (plan["trajectory"]["seed"], eid, role)
                record = collect_one(env, observation, plan, infos[key], args.output)
                temp = record_path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, record_path)
                completed += 1
                if completed % 20 == 0:
                    print(f"replayed {completed}/{len(episodes)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": record_id, "error": repr(exc)})
                print(f"ERROR {record_id}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete replay queue")
    summary = {"schema": "full_history_collection_v1",
               "probe_manifest_sha256": digest(args.probe_manifest),
               "requested_pairs": len(selected), "split": args.split,
               "completed_trajectories": completed,
               "errors": errors}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or completed != len(episodes):
        raise RuntimeError(f"replay coverage {completed}/{len(episodes)}")


if __name__ == "__main__":
    main()
