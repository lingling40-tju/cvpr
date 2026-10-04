"""Replay frozen n=4 control actions for geodesic labels without RGB writes.

Only selected R2R-train fit IDs are accessed. This delegates the action
execution and terminal-distance drift check to the audited visual replayer,
but replaces its frame writer with a no-op and persists label-only records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import collect_policy_process_turns as replay


EXPECTED_IDS_SHA = "d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081"


def label_record(plan: dict, source: dict, manifest_sha: str) -> dict:
    distances = [source["start_distance_to_goal_for_label_only"]] + [
        turn["distance_to_goal_for_label_only"] for turn in source["turns"]]
    if not all(math.isfinite(float(value)) and value >= 0
               for value in distances):
        raise ValueError("nonfinite replay geodesic label")
    forward = sum(a - b >= 1.0 for a, b in zip(distances[:-1], distances[1:]))
    backward = sum(b - a >= 1.0 for a, b in zip(distances[:-1], distances[1:]))
    if forward != source["progress_turns_geodesic_1m"] or \
            backward != source["regression_turns_geodesic_1m"]:
        raise ValueError("replay label count mismatch")
    return {
        "schema": "control_fit_label_only_replay_v1",
        "manifest_sha256": manifest_sha,
        "record_id": source["record_id"],
        "seed": int(plan["seed"]),
        "episode_id": str(plan["episode_id"]),
        "variant": int(plan["variant"]),
        "scene_id": plan["scene_id"],
        "terminal_mode": plan["terminal_mode"],
        "source_terminal_distance_m":
            plan["terminal_distance_m_for_replay_audit_only"],
        "replayed_terminal_distance_m":
            source["replayed_terminal_distance_m_for_audit_only"],
        "turn_indices": [turn["original_turn_index"] for turn in source["turns"]],
        "distance_to_goal_m_for_fit_label_only": distances,
        "forward_turns_1m": forward,
        "regression_turns_1m": backward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("negative smoke limit")
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = replay.digest(args.manifest)
    if manifest["schema"] != "policy_process_train_manifest_v1" or \
            manifest["source_id_manifest_sha256"] != EXPECTED_IDS_SHA or \
            manifest["targets"] != {"fit": 1024, "development": 0,
                                    "audit": 0} or \
            manifest["selected"]["development"] or \
            manifest["selected"]["audit"] or \
            replay.digest(replay.DATASET) != manifest["train_dataset_sha256"]:
        raise ValueError("label-only fit source changed")
    plans = manifest["selected"]["fit"][:args.limit or None]
    if len(manifest["selected"]["fit"]) != 1024 or \
            len({(str(p["episode_id"]), int(p["variant"])) for p in plans}) \
            != len(plans):
        raise ValueError("fit variants incomplete or repeated")
    infos = replay.source_infos(manifest, plans)
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
    episodes_by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    if len(episodes_by_id) != len(dataset.episodes):
        raise ValueError("ambiguous R2R train episode ID")
    plans.sort(key=lambda p: (p["scene_id"], str(p["episode_id"]),
                              int(p["variant"])))
    episodes = []
    for plan in plans:
        eid = str(plan["episode_id"])
        episode = episodes_by_id.get(eid)
        if episode is None or \
                replay.canonical_scene(str(episode.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing R2R train episode {eid}")
        episodes.append(episode)
    dataset.episodes = episodes
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir(exist_ok=True)
    # Geodesic supervision is available from Habitat; no RGB is needed for
    # this pass. The visual replay's terminal check is retained verbatim.
    replay.save_frame = lambda observation, path: None
    started = time.time()
    completed = []
    with replay.Env(config.TASK_CONFIG, dataset=dataset) as env:
        for plan in plans:
            observation = env.reset()
            eid = str(plan["episode_id"])
            if str(env.current_episode.episode_id) != eid:
                raise RuntimeError("Habitat episode reset order changed")
            rid = replay.record_id(plan)
            path = args.output / "records" / f"{rid}.json"
            if path.is_file():
                old = json.loads(path.read_text())
                if old.get("schema") != "control_fit_label_only_replay_v1" or \
                        old.get("manifest_sha256") != manifest_sha or \
                        old.get("record_id") != rid or \
                        old.get("scene_id") != plan["scene_id"] or \
                        abs(old.get("source_terminal_distance_m", -1) -
                            plan["terminal_distance_m_for_replay_audit_only"]) > 1e-5:
                    raise ValueError(f"stale replay record {rid}")
                completed.append(old)
                continue
            info = infos[(11, eid, int(plan["variant"]))]
            raw = replay.collect_one(env, observation, plan, info,
                                     args.output, manifest_sha)
            record = label_record(plan, raw, manifest_sha)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(temp, path)
            completed.append(record)
            if len(completed) % 20 == 0:
                print(f"label-only {len(completed)}/{len(plans)}", flush=True)
    summary = {
        "schema": "control_fit_label_only_replay_summary_v1",
        "manifest_sha256": manifest_sha,
        "requested": len(plans), "completed": len(completed),
        "unique_episode_ids": len({row["episode_id"] for row in completed}),
        "forward_turns_1m": sum(row["forward_turns_1m"] for row in completed),
        "regression_turns_1m": sum(row["regression_turns_1m"]
                                   for row in completed),
        "episode_ids_with_regression": len({row["episode_id"] for row in
                                            completed if row["regression_turns_1m"]}),
        "terminal_modes": dict(Counter(row["terminal_mode"] for row in completed)),
        "rgb_frames_written": 0,
        "smoke_limit": args.limit,
        "elapsed_seconds": time.time() - started,
    }
    if len(completed) != len(plans):
        raise RuntimeError("label-only replay incomplete")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
