"""Freeze a reachable wrong goal per fit episode using only its start state.

This preflight takes no policy actions, reads no intermediate distance labels,
and never opens development/audit/val-unseen episodes. It only resolves the
ordered train-only candidate list against Habitat's static navigation mesh.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import collect_policy_process_turns as replay
from prepare_cross_goal_fit_manifest import RENDER_SHA, digest, goal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text())
    if (candidates["schema"] != "cross_goal_fit_candidate_manifest_v1"
            or candidates["source_sha256"]["render_manifest"] != RENDER_SHA
            or candidates["episode_ids"] < 200):
        raise ValueError("candidate source invalid")
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
        raise ValueError("ambiguous R2R train ID")
    scene_by_id = {str(p["episode_id"]): p["scene_id"]
                   for p in candidates["plans"]}
    source_ids = sorted(candidates["ranked_candidate_episode_ids"],
                        key=lambda eid: (scene_by_id[eid], eid))
    episodes = [by_id[eid] for eid in source_ids]
    dataset.episodes = episodes
    chosen = {}
    no_reachable = []
    started = time.time()
    with replay.Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in episodes:
            env.reset()
            eid = str(env.current_episode.episode_id)
            if eid in chosen or eid in no_reachable or eid not in scene_by_id:
                raise RuntimeError(f"unexpected Habitat episode reset {eid}")
            scene = replay.canonical_scene(str(env.current_episode.scene_id))
            if scene != scene_by_id[eid]:
                raise ValueError(f"crossed-goal source scene changed {eid}")
            position = env.sim.get_agent_state().position
            own_distance = float(env.get_metrics()["distance_to_goal"])
            if not math.isfinite(own_distance):
                raise ValueError(f"unreachable correct goal {eid}")
            for oid in candidates["ranked_candidate_episode_ids"][eid]:
                other = by_id[oid]
                if replay.canonical_scene(str(other.scene_id)) != scene:
                    raise ValueError(f"wrong candidate scene changed {eid}/{oid}")
                distance = float(env.sim.geodesic_distance(
                    position, goal({"goals": other.goals})))
                if math.isfinite(distance) and distance >= 0:
                    chosen[eid] = {"wrong_episode_id": oid,
                                   "wrong_goal_position_for_label_only":
                                       goal({"goals": other.goals}),
                                   "wrong_instruction":
                                       other.instruction.instruction_text.strip(),
                                   "wrong_start_distance_m_for_selection_only":
                                       distance}
                    break
            if eid not in chosen:
                no_reachable.append(eid)
            if (len(chosen) + len(no_reachable)) % 25 == 0:
                print(f"reachability {len(chosen)+len(no_reachable)}/{len(source_ids)}",
                      flush=True)
    if len(chosen) + len(no_reachable) != len(source_ids):
        raise RuntimeError("reachability coverage incomplete")
    plans = []
    for provisional in candidates["plans"]:
        eid = provisional["episode_id"]
        if eid in chosen:
            plan = {k: provisional[k] for k in
                    ("seed", "episode_id", "variant", "scene_id")}
            plan.update(chosen[eid])
            plans.append(plan)
    counts = Counter(p["episode_id"] for p in plans)
    if (len(chosen) < 200 or len(plans) != 2 * len(chosen)
            or set(counts.values()) != {2}):
        raise ValueError("too few reachable fit pairs")
    report = {"schema": "cross_goal_fit_manifest_v1",
              "interpretation": "Train-only reachable wrong goals frozen before motion labels; coordinates never enter model input",
              "source_sha256": {
                  **candidates["source_sha256"],
                  "candidate_manifest": digest(args.candidates)},
              "selection": "first finite start-state geodesic in frozen farthest-goal order; no policy action taken",
              "episode_ids": len(chosen), "trajectories": len(plans),
              "skipped_no_distinct_goal": candidates["skipped_episode_ids"],
              "skipped_no_reachable_goal": sorted(no_reachable),
              "elapsed_seconds": time.time() - started,
              "plans": plans}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reachable_ids": len(chosen), "plans": len(plans),
                      "unreachable_ids": len(no_reachable),
                      "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
