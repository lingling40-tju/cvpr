"""Replay cached fit actions for two-goal geodesic labels, without RGB writes.

The correct-goal trace must agree with the independently rendered record at
every turn. Wrong-goal coordinates are used only inside the simulator. This
collector does not run a policy or open development/audit/val-unseen data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
import os
from pathlib import Path
import time

import collect_policy_process_turns as replay
from prepare_cross_goal_fit_manifest import RENDER_SHA, digest


MIN_CROSSED_TURNS = 100
MIN_CROSSED_EPISODES = 50
CHANGE_M = 0.5


def cross_count(correct: list[float], wrong: list[float]) -> int:
    if len(correct) != len(wrong):
        raise ValueError("goal traces differ in length")
    return sum((c0 - c1 >= CHANGE_M and w0 - w1 <= -CHANGE_M)
               or (c0 - c1 <= -CHANGE_M and w0 - w1 >= CHANGE_M)
               for c0, c1, w0, w1 in zip(
                   correct[:-1], correct[1:], wrong[:-1], wrong[1:]))


def read_render_record(plan: dict, render_root: Path) -> dict:
    rid = replay.record_id(plan)
    path = render_root / "fit" / "records" / f"{rid}.json"
    record = json.loads(path.read_text())
    if (record["schema"] != "policy_process_turn_record_v1"
            or record["manifest_sha256"] != RENDER_SHA
            or record["record_id"] != rid
            or str(record["episode_id"]) != plan["episode_id"]
            or record["scene_id"] != plan["scene_id"]
            or not record["turns"]):
        raise ValueError(f"render source mismatch {rid}")
    return record


def distance_to_wrong(env, target: list[float]) -> float:
    position = env.sim.get_agent_state().position
    result = float(env.sim.geodesic_distance(position, target))
    if not math.isfinite(result) or result < 0:
        raise ValueError("nonfinite crossed-goal geodesic")
    return result


def collect_one(env, plan: dict, rendered: dict,
                manifest_sha: str) -> dict:
    eid = plan["episode_id"]
    if (str(env.current_episode.episode_id) != eid
            or replay.canonical_scene(str(env.current_episode.scene_id))
            != plan["scene_id"]):
        raise ValueError(f"Habitat episode/scene mismatch {eid}")
    if (rendered["instruction"].strip()
            != env.current_episode.instruction.instruction_text.strip()):
        raise ValueError(f"correct instruction mismatch {eid}")
    correct = [float(env.get_metrics()["distance_to_goal"])]
    wrong = [distance_to_wrong(
        env, plan["wrong_goal_position_for_label_only"])]
    expected = [float(rendered["start_distance_to_goal_for_label_only"])] + [
        float(row["distance_to_goal_for_label_only"])
        for row in rendered["turns"]]
    if abs(correct[0] - expected[0]) > 1e-4:
        raise ValueError(f"initial correct-goal distance drift {eid}")
    for turn in rendered["turns"]:
        for action in turn["motion_actions"]:
            code, repeat = replay.action_steps(action)
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError(f"early replay termination {eid}")
                env.step({"action": code})
        correct.append(float(env.get_metrics()["distance_to_goal"]))
        wrong.append(distance_to_wrong(
            env, plan["wrong_goal_position_for_label_only"]))
    drift = max(abs(a - b) for a, b in zip(correct, expected))
    if len(correct) != len(expected) or drift > 1e-4:
        raise ValueError(f"correct-goal turn replay drift {eid}: {drift}")
    crossed = cross_count(correct, wrong)
    return {"schema": "cross_goal_fit_label_record_v1",
            "manifest_sha256": manifest_sha,
            "render_manifest_sha256": RENDER_SHA,
            "record_id": rendered["record_id"],
            "episode_id": eid, "variant": plan["variant"],
            "scene_id": plan["scene_id"],
            "wrong_episode_id": plan["wrong_episode_id"],
            "correct_distance_m_for_label_only": correct,
            "wrong_distance_m_for_label_only": wrong,
            "crossed_turns_0p5m": crossed,
            "max_correct_distance_drift_m": drift}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0 or digest(args.render_manifest) != RENDER_SHA:
        raise ValueError("invalid limit or changed render source")
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    if (manifest["schema"] != "cross_goal_fit_manifest_v1"
            or manifest["source_sha256"]["render_manifest"] != RENDER_SHA
            or manifest["episode_ids"] < 200
            or manifest["trajectories"] != len(manifest["plans"])):
        raise ValueError("cross-goal manifest invalid")
    plans = list(manifest["plans"][:args.limit or None])
    if len({(p["episode_id"], p["variant"]) for p in plans}) != len(plans):
        raise ValueError("duplicate selected trajectory")
    from VLN_CE.vlnce_baselines.config.default import get_config
    config = get_config(replay.CONFIG)
    config.defrost()
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "train"
    config.TASK_CONFIG.TASK.NDTW.SPLIT = "train"
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL"]
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.TASK_CONFIG.freeze()
    config.freeze()
    dataset = replay.habitat.datasets.make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    if len(by_id) != len(dataset.episodes):
        raise ValueError("ambiguous R2R train episode IDs")
    plans.sort(key=lambda p: (p["scene_id"], p["episode_id"], p["variant"]))
    episodes = []
    requested = defaultdict(deque)
    for plan in plans:
        eid = plan["episode_id"]
        ep = by_id.get(eid)
        if ep is None or replay.canonical_scene(str(ep.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing fit episode {eid}")
        episodes.append(ep)
        requested[eid].append(plan)
    dataset.episodes = episodes
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir(exist_ok=True)
    completed = []
    started = time.time()
    with replay.Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in plans:
            env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected Habitat reset {eid}")
            plan = requested[eid].popleft()
            rendered = read_render_record(plan, args.render_root)
            path = args.output / "records" / f"{rendered['record_id']}.json"
            if path.is_file():
                record = json.loads(path.read_text())
                if (record.get("schema") != "cross_goal_fit_label_record_v1"
                        or record.get("manifest_sha256") != manifest_sha
                        or record.get("record_id") != rendered["record_id"]
                        or record.get("wrong_episode_id") != plan["wrong_episode_id"]):
                    raise ValueError(f"stale crossed-goal record {path.name}")
            else:
                record = collect_one(env, plan, rendered, manifest_sha)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(temp, path)
            completed.append(record)
            if len(completed) % 25 == 0:
                print(f"cross-goal {len(completed)}/{len(plans)}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete Habitat reset coverage")
    crossed_ids = {row["episode_id"] for row in completed
                   if row["crossed_turns_0p5m"]}
    summary = {
        "schema": "cross_goal_fit_label_summary_v1",
        "manifest_sha256": manifest_sha,
        "requested": len(plans), "completed": len(completed),
        "unique_episode_ids": len({row["episode_id"] for row in completed}),
        "crossed_turns_0p5m": sum(row["crossed_turns_0p5m"]
                                   for row in completed),
        "episode_ids_with_crossed_turns": len(crossed_ids),
        "max_correct_distance_drift_m": max(
            row["max_correct_distance_drift_m"] for row in completed),
        "smoke_limit": args.limit,
        "elapsed_seconds": time.time() - started,
        "sample_gate": {"minimum_crossed_turns": MIN_CROSSED_TURNS,
                        "minimum_episode_ids": MIN_CROSSED_EPISODES,
                        "passed": (sum(row["crossed_turns_0p5m"]
                                       for row in completed) >= MIN_CROSSED_TURNS
                                   and len(crossed_ids) >= MIN_CROSSED_EPISODES)
                        if not args.limit else None},
        "terminal_modes": dict(Counter(
            read_render_record(p, args.render_root)["terminal_mode"]
            for p in plans)),
    }
    if len(completed) != len(plans):
        raise RuntimeError("crossed-goal label coverage incomplete")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
