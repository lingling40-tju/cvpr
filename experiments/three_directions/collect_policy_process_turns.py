"""Replay frozen group-four R2R-train rollouts at every executed turn.

RGB and action history are model inputs. Simulator geodesic distances are
stored in separate label fields, solely for training and auditing progress.
The timeout's 13th, unexecuted response is ignored. Scene-disjoint subsets
can run on separate Habitat GPUs and resume from complete records.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
import os
from pathlib import Path

import habitat
from habitat import Env

from collect_policy_preference_frames import (
    CONFIG, DATASET, action_steps, canonical_scene, digest, save_frame,
)


def record_id(plan: dict) -> str:
    return f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"


def source_infos(manifest: dict, plans: list[dict]) -> dict[tuple[int, str, int], dict]:
    requested = {(plan["seed"], str(plan["episode_id"])) for plan in plans}
    result = {}
    for seed in sorted({seed for seed, _ in requested}):
        source = manifest["sources"][str(seed)]
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError(f"rollout checksum mismatch seed {seed}")
        groups = defaultdict(list)
        with path.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    eid = str(info["episode_id"])
                    if (seed, eid) in requested:
                        groups[eid].append(info)
        for plan in (plan for plan in plans if plan["seed"] == seed):
            eid, variant = str(plan["episode_id"]), plan["variant"]
            if len(groups[eid]) != 4:
                raise ValueError(f"missing group-four source: {seed}/{eid}")
            info = groups[eid][variant]
            if info["end_reason"] != plan["terminal_mode"] or \
                    abs(float(info["distance_to_goal"]) -
                        plan["terminal_distance_m_for_replay_audit_only"]) > 1e-5:
                raise ValueError(f"source/manifest mismatch: {seed}/{eid}/{variant}")
            result[(seed, eid, variant)] = info
    if len(result) != len(plans):
        raise ValueError("source lookup coverage mismatch")
    return result


def collect_one(env: Env, observation: dict, plan: dict, info: dict,
                output: Path, manifest_sha: str) -> dict:
    eid = str(plan["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != plan["scene_id"] or \
            observation["instruction"]["text"].strip() != info["instruction"].strip():
        raise ValueError(f"replay episode/scene/instruction mismatch: {eid}")
    rid = record_id(plan)
    folder = output / "frames" / rid
    folder.mkdir(parents=True, exist_ok=True)
    initial = folder / "0000.jpg"
    save_frame(observation, initial)
    start_distance = float(env.get_metrics()["distance_to_goal"])
    if not math.isfinite(start_distance):
        raise ValueError("nonfinite start distance")
    turns = []
    executed_count = 0
    stop_seen = False
    for original_index, turn in enumerate(info["gen_traj"], 1):
        executed = list(turn.get("executed_actions", []))
        if not executed:
            if original_index != len(info["gen_traj"]):
                raise ValueError("empty nonterminal executed turn")
            continue
        if stop_seen or "stop" in executed[:-1]:
            raise ValueError("action after STOP or early STOP")
        motion = [action for action in executed if action != "stop"]
        for action in motion:
            code, repeat = action_steps(action)
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError(f"early replay termination {rid}")
                observation = env.step({"action": code})
            executed_count += 1
        stop_seen = "stop" in executed
        if motion:
            # A mixed motion+STOP response is truncated after executed motion.
            response = ", ".join(motion) if stop_seen else turn["response"]
            if "stop" in response.lower():
                raise ValueError("terminal STOP in model history")
            frame = folder / f"{len(turns) + 1:04d}.jpg"
            save_frame(observation, frame)
            distance = float(env.get_metrics()["distance_to_goal"])
            if not math.isfinite(distance):
                raise ValueError("nonfinite turn distance")
            turns.append({"original_turn_index": original_index,
                          "assistant_response": response,
                          "motion_actions": motion,
                          "image": str(frame.relative_to(output)),
                          "distance_to_goal_for_label_only": distance})
    source_distance = plan["terminal_distance_m_for_replay_audit_only"]
    terminal_distance = float(env.get_metrics()["distance_to_goal"])
    if not turns or abs(terminal_distance - source_distance) > .25:
        raise RuntimeError(f"terminal replay drift {rid}: {terminal_distance:.3f} vs "
                           f"{source_distance:.3f}")
    deltas = [after - before for before, after in zip(
        [start_distance] + [t["distance_to_goal_for_label_only"] for t in turns[:-1]],
        [t["distance_to_goal_for_label_only"] for t in turns])]
    return {"schema": "policy_process_turn_record_v1",
            "manifest_sha256": manifest_sha,
            "record_id": rid, "seed": plan["seed"], "episode_id": eid,
            "variant": plan["variant"], "scene_id": plan["scene_id"],
            "instruction": info["instruction"].strip(),
            "terminal_mode": plan["terminal_mode"],
            "initial_image": str(initial.relative_to(output)),
            "turns": turns, "motion_actions": executed_count,
            "terminal_stop_response_omitted": stop_seen,
            "start_distance_to_goal_for_label_only": start_distance,
            "replayed_terminal_distance_m_for_audit_only": terminal_distance,
            "source_terminal_distance_m_for_audit_only": source_distance,
            "progress_turns_geodesic_1m": sum(delta <= -1.0 for delta in deltas),
            "regression_turns_geodesic_1m": sum(delta >= 1.0 for delta in deltas)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("negative limit")
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "policy_process_train_manifest_v1" or \
            digest(DATASET) != manifest["train_dataset_sha256"]:
        raise ValueError("manifest or train dataset checksum mismatch")
    plans = manifest["selected"][args.part][:args.limit or None]
    if not plans:
        raise ValueError("empty selection")
    identities = [(p["seed"], str(p["episode_id"]), p["variant"]) for p in plans]
    if len(set(identities)) != len(plans):
        raise ValueError("duplicate selected trajectory")
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
    by_id = {str(episode.episode_id): episode for episode in dataset.episodes}
    if len(by_id) != len(dataset.episodes):
        raise ValueError("ambiguous train episode IDs")
    plans.sort(key=lambda p: (p["scene_id"], str(p["episode_id"]),
                              p["seed"], p["variant"]))
    requested = defaultdict(deque)
    episodes = []
    for plan in plans:
        eid = str(plan["episode_id"])
        episode = by_id.get(eid)
        if episode is None or canonical_scene(str(episode.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing Habitat episode {eid}")
        episodes.append(episode)
        requested[eid].append(plan)
    dataset.episodes = episodes
    output = args.output_root / args.part
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = digest(args.manifest)
    completed, errors = [], []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in episodes:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected Habitat reset {eid}")
            plan = requested[eid].popleft()
            rid = record_id(plan)
            record_path = output / "records" / f"{rid}.json"
            record_path.parent.mkdir(exist_ok=True)
            if record_path.is_file():
                prior = json.loads(record_path.read_text())
                if prior["manifest_sha256"] == manifest_sha and \
                        (output / prior["initial_image"]).is_file() and \
                        all((output / turn["image"]).is_file() for turn in prior["turns"]):
                    completed.append(prior)
                    continue
            try:
                info = infos[(plan["seed"], eid, plan["variant"])]
                record = collect_one(env, observation, plan, info, output, manifest_sha)
                temp = record_path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, record_path)
                completed.append(record)
                if len(completed) % 20 == 0:
                    print(f"{args.part} {len(completed)}/{len(episodes)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": rid, "error": repr(exc)})
                print(f"ERROR {rid}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete Habitat reset coverage")
    summary = {"schema": "policy_process_turn_collection_v1",
               "part": args.part, "manifest_sha256": manifest_sha,
               "requested": len(episodes), "completed": len(completed),
               "unique_episodes": len({r["episode_id"] for r in completed}),
               "terminal_modes": dict(Counter(r["terminal_mode"] for r in completed)),
               "progress_turns_geodesic_1m": sum(r["progress_turns_geodesic_1m"]
                                                 for r in completed),
               "regression_turns_geodesic_1m": sum(r["regression_turns_geodesic_1m"]
                                                   for r in completed),
               "errors": errors, "smoke_limit": args.limit}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or len(completed) != len(episodes):
        raise RuntimeError(f"incomplete replay {len(completed)}/{len(episodes)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
