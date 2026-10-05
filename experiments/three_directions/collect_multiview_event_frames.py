"""Replay frozen train trajectories and collect only needed two-view frames."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import shutil

from collect_boundary_occupancy_frames import (shard_for, source_infos,
                                               write_json)
from collect_policy_preference_frames import (
    CONFIG, DATASET, action_steps, canonical_scene, digest, save_frame,
)
from preflight_boundary_occupancy_source import DATA_SHA


BOUNDARY_SHA = "179f726b2f91dd269238559ddc71c35e9c77dbfa1ebd287f51b4e73a7b134b10"


def old_frame_index(manifest: dict, old_root: Path, part: str) -> dict:
    result = {}
    for plan in manifest["selected"][part]:
        rid = plan["record_id"]
        record = json.loads((old_root / part / "records" /
                             f"{rid}.json").read_text())
        if record["manifest_sha256"] != BOUNDARY_SHA or \
                record["record_id"] != rid:
            raise ValueError(f"changed verified crossing RGB: {rid}")
        for role, index in (("outside", plan["outside_state_index"]),
                            ("inside", plan["inside_state_index"])):
            path = old_root / part / record["input"]["images"][role]
            if not path.is_file():
                raise ValueError(f"missing verified crossing frame: {rid}")
            result[(rid, index)] = path
    return result


def collect_one(env, observation: dict, plan: dict, info: dict,
                output: Path, manifest_sha: str, reused: dict) -> tuple[dict, dict]:
    rid = plan["record_id"]
    eid = str(plan["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != plan["scene_id"] or \
            observation["instruction"]["text"].strip() != plan["instruction"]:
        raise ValueError(f"Habitat episode or instruction changed: {rid}")
    selected = set(plan["states_to_capture"])
    if not selected or min(selected) < 0 or \
            max(selected) > len(info["gen_traj"]):
        raise ValueError(f"invalid selected states: {rid}")
    folder = output / "frames" / rid
    folder.mkdir(parents=True, exist_ok=True)
    images, states, copied = {}, {}, 0

    def capture(index: int, view: dict, distance: float) -> None:
        nonlocal copied
        if index not in selected:
            return
        frame = folder / f"{index:04d}.jpg"
        old = reused.get((rid, index))
        if old is None:
            save_frame(view, frame)
        else:
            shutil.copyfile(old, frame)
            copied += 1
        images[str(index)] = str(frame.relative_to(output))
        states[str(index)] = distance

    start = float(env.get_metrics()["distance_to_goal"])
    if not math.isfinite(start) or \
            abs(start-float(info["oracle_start_distance"])) > .25:
        raise RuntimeError(f"start geodesic drift: {rid}")
    capture(0, observation, start)
    before = start
    stop_seen = False
    for turn_index, turn in enumerate(info["gen_traj"], 1):
        actions = list(turn.get("executed_actions", []))
        if stop_seen or "stop" in actions[:-1]:
            raise ValueError(f"action after STOP: {rid}/{turn_index}")
        for action in actions:
            if action == "stop":
                continue
            code, repeat = action_steps(action)
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError(f"early replay termination: {rid}")
                observation = env.step({"action": code})
        stop_seen = "stop" in actions
        after = float(env.get_metrics()["distance_to_goal"])
        if not math.isfinite(after) or \
                abs(before-float(turn["oracle_before_distance"])) > .25 or \
                abs(after-float(turn["oracle_after_distance"])) > .25:
            raise RuntimeError(f"turn geodesic drift: {rid}/{turn_index}")
        capture(turn_index, observation, after)
        before = after
    if set(images) != {str(i) for i in selected} or \
            abs(before-float(info["distance_to_goal"])) > .25:
        raise RuntimeError(f"incomplete state or terminal replay: {rid}")
    record = {
        "schema": "multiview_event_rgb_input_v1",
        "capture_manifest_sha256": manifest_sha,
        "record_id": rid,
        "input": {
            "instruction": plan["instruction"],
            "terminal_clause": plan["terminal_clause"],
            "wrong_instruction": plan["wrong_instruction"],
            "wrong_terminal_clause": plan["wrong_terminal_clause"],
            "images_by_state": images,
        },
    }
    audit = {
        "schema": "multiview_event_rgb_replay_audit_v1",
        "capture_manifest_sha256": manifest_sha,
        "record_id": rid, "seed": plan["seed"],
        "episode_id": eid, "variant": plan["variant"],
        "scene_id": plan["scene_id"],
        "state_distance_m": states,
        "source_state_distance_m": {str(i): float(
            info["oracle_start_distance"] if i == 0 else
            info["gen_traj"][i-1]["oracle_after_distance"])
            for i in selected},
        "source_terminal_distance_m": float(info["distance_to_goal"]),
        "replay_terminal_distance_m": before,
        "copied_verified_crossing_images": copied,
    }
    return record, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--old-rgb-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--record-id", default="")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards <= 4 or \
            args.record_id and args.shards != 1:
        raise ValueError("invalid replay shard")
    manifest = json.loads(args.capture_manifest.read_text())
    report = json.loads(args.source_report.read_text())
    old = json.loads(args.boundary_manifest.read_text())
    manifest_sha = digest(args.capture_manifest)
    if manifest["schema"] != "multiview_event_rgb_capture_manifest_v1" or \
            manifest["group_size"] != 4 or manifest["seeds"] != [11, 22, 33] or \
            manifest["boundary_manifest_sha256"] != \
                digest(args.boundary_manifest) != BOUNDARY_SHA or \
            report["capture_manifest_sha256"] != manifest_sha or \
            not report["ready_for_rgb_replay"] or report["audit_selected"] or \
            digest(DATASET) != DATA_SHA:
        raise ValueError("changed or invalid n4 event source")
    plans = [p for p in manifest["selected"][args.part] if
             shard_for(p, args.shards) == args.shard]
    if args.record_id:
        plans = [p for p in plans if p["record_id"] == args.record_id]
    if not plans or len({p["record_id"] for p in plans}) != len(plans):
        raise ValueError("empty or duplicate capture selection")
    infos = source_infos(manifest, args.root, plans)
    reuse = old_frame_index(old, args.old_rgb_root, args.part)

    import habitat
    from habitat import Env
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
    plans.sort(key=lambda p: (p["scene_id"], str(p["episode_id"]),
                              p["seed"], p["variant"]))
    requested = defaultdict(deque)
    selected = []
    for plan in plans:
        eid = str(plan["episode_id"])
        ep = by_id.get(eid)
        if ep is None or canonical_scene(str(ep.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing train episode: {eid}")
        selected.append(ep)
        requested[eid].append(plan)
    dataset.episodes = selected
    output = args.output_root / args.part
    output.mkdir(parents=True, exist_ok=True)
    completed = resumed = 0
    errors = []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected Habitat reset: {eid}")
            plan = requested[eid].popleft()
            rid = plan["record_id"]
            record_path = output / "records" / f"{rid}.json"
            audit_path = output / "audits" / f"{rid}.json"
            if record_path.is_file() and audit_path.is_file():
                record = json.loads(record_path.read_text())
                audit = json.loads(audit_path.read_text())
                if record.get("capture_manifest_sha256") == manifest_sha and \
                        audit.get("capture_manifest_sha256") == manifest_sha and \
                        record.get("record_id") == audit.get("record_id") == rid and \
                        set(record.get("input", {}).get("images_by_state", {})) == \
                        {str(i) for i in plan["states_to_capture"]} and \
                        all((output / rel).is_file() for rel in
                            record["input"]["images_by_state"].values()):
                    completed += 1
                    resumed += 1
                    continue
            try:
                info = infos[(plan["seed"], eid, plan["variant"])]
                record, audit = collect_one(env, observation, plan, info,
                                            output, manifest_sha, reuse)
                write_json(audit_path, audit)
                write_json(record_path, record)
                completed += 1
                if completed % 50 == 0:
                    print(f"{args.part} {completed}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": rid, "error": repr(exc)})
                print(f"ERROR {rid}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete Habitat reset coverage")
    summary = {
        "schema": "multiview_event_rgb_collection_v1",
        "part": args.part, "capture_manifest_sha256": manifest_sha,
        "requested": len(selected), "completed": completed,
        "unique_episode_ids": len({p["episode_id"] for p in plans}),
        "state_images_expected": sum(len(p["states_to_capture"])
                                     for p in plans),
        "resumed": resumed, "errors": errors,
        "shard": args.shard, "shards": args.shards,
        "target_record_id": args.record_id,
    }
    name = "summary.json" if args.shards == 1 else \
           f"summary.shard{args.shard}.json"
    write_json(output / name, summary)
    if errors or completed != len(selected):
        raise RuntimeError(f"incomplete event replay {completed}/{len(selected)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
