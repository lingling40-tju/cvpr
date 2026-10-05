"""Replay selected n=4 train trajectories and capture two boundary views.

Only the nested `input` in each record may be passed to a visual model.
Privileged geodesic replay audits are saved in separate files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

from collect_policy_preference_frames import (
    CONFIG, DATASET, action_steps, canonical_scene, digest, save_frame,
)
from preflight_boundary_occupancy_source import DATA_SHA


def shard_for(plan: dict, shards: int) -> int:
    key = f"{plan['seed']}:{plan['episode_id']}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % shards


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_infos(manifest: dict, root: Path, plans: list[dict]) -> dict:
    wanted = {(plan["seed"], str(plan["episode_id"])) for plan in plans}
    groups = defaultdict(list)
    for seed in sorted({seed for seed, _ in wanted}):
        rollout = root / f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
        expected = manifest["source_sha256"]["seeds"][str(seed)]
        audit = root / f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json"
        if digest(rollout) != expected["rollout"] or \
                digest(audit) != expected["train_audit"]:
            raise ValueError(f"audited source changed: seed {seed}")
        with rollout.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    eid = str(info["episode_id"])
                    if (seed, eid) in wanted:
                        groups[(seed, eid)].append(info)
    result = {}
    for plan in plans:
        seed, eid, variant = plan["seed"], str(plan["episode_id"]), plan["variant"]
        four = groups[(seed, eid)]
        if len(four) != 4 or not 0 <= variant < 4:
            raise ValueError(f"changed n4 source group: {seed}/{eid}")
        info = four[variant]
        if info["instruction"].strip() != plan["instruction"]:
            raise ValueError(f"source instruction changed: {seed}/{eid}")
        result[(seed, eid, variant)] = info
    if len(result) != len(plans):
        raise ValueError("source plan coverage mismatch")
    return result


def collect_one(env, observation: dict, plan: dict, info: dict,
                output: Path, manifest_sha: str) -> tuple[dict, dict]:
    eid = str(plan["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != plan["scene_id"] or \
            observation["instruction"]["text"].strip() != plan["instruction"]:
        raise ValueError(f"Habitat episode/scene/instruction mismatch: {eid}")
    outside = plan["outside_state_index"]
    inside = plan["inside_state_index"]
    if not isinstance(outside, int) or not isinstance(inside, int) or \
            not 0 <= outside < inside <= outside + 2 or \
            inside > len(info["gen_traj"]):
        raise ValueError("invalid frozen boundary state indices")
    rid = plan["record_id"]
    folder = output / "frames" / rid
    folder.mkdir(parents=True, exist_ok=True)
    indices = {outside, inside}
    images = {}
    histories = {}
    audit_states = {}
    history = []
    start_distance = float(env.get_metrics()["distance_to_goal"])
    if not math.isfinite(start_distance) or \
            abs(start_distance - float(info["oracle_start_distance"])) > .25:
        raise RuntimeError(f"start distance drift: {rid}")

    def capture(state_index: int, view: dict, distance: float) -> None:
        if state_index not in indices:
            return
        frame = folder / f"{state_index:04d}.jpg"
        save_frame(view, frame)
        images[str(state_index)] = str(frame.relative_to(output))
        histories[str(state_index)] = list(history)
        audit_states[str(state_index)] = distance

    capture(0, observation, start_distance)
    before_distance = start_distance
    stop_seen = False
    for turn_index, turn in enumerate(info["gen_traj"], 1):
        executed = list(turn.get("executed_actions", []))
        if stop_seen or "stop" in executed[:-1]:
            raise ValueError(f"action after STOP: {rid}/{turn_index}")
        motion = [action for action in executed if action != "stop"]
        for action in motion:
            code, repeats = action_steps(action)
            for _ in range(repeats):
                if env.episode_over:
                    raise RuntimeError(f"early replay termination: {rid}")
                observation = env.step({"action": code})
        stop_seen = "stop" in executed
        after_distance = float(env.get_metrics()["distance_to_goal"])
        if not math.isfinite(after_distance) or \
                abs(before_distance - float(turn["oracle_before_distance"])) > .25 or \
                abs(after_distance - float(turn["oracle_after_distance"])) > .25:
            raise RuntimeError(f"turn geodesic drift: {rid}/{turn_index}")
        if motion:
            history.append({"turn": turn_index, "executed_actions": motion})
        capture(turn_index, observation, after_distance)
        before_distance = after_distance
    if set(images) != {str(outside), str(inside)}:
        raise RuntimeError(f"boundary RGB missing: {rid}")
    source_terminal = float(info["distance_to_goal"])
    if abs(before_distance - source_terminal) > .25:
        raise RuntimeError(f"terminal distance drift: {rid}")
    input_value = {
        "instruction": plan["instruction"],
        "wrong_instruction": plan["wrong_instruction"],
        "images": {"outside": images[str(outside)],
                   "inside": images[str(inside)]},
        "action_history_by_state": {
            "outside": histories[str(outside)],
            "inside": histories[str(inside)],
        },
    }
    record = {
        "schema": "boundary_occupancy_rgb_model_input_v1",
        "manifest_sha256": manifest_sha,
        "record_id": rid, "input": input_value,
    }
    audit = {
        "schema": "boundary_occupancy_rgb_replay_audit_v1",
        "manifest_sha256": manifest_sha,
        "record_id": rid, "seed": plan["seed"],
        "episode_id": eid, "variant": plan["variant"],
        "scene_id": plan["scene_id"],
        "state_distance_m": audit_states,
        "source_outside_distance_m":
            float(info["oracle_start_distance"] if outside == 0 else
                  info["gen_traj"][outside - 1]["oracle_after_distance"]),
        "source_inside_distance_m":
            float(info["gen_traj"][inside - 1]["oracle_after_distance"]),
        "terminal_distance_m": before_distance,
        "source_terminal_distance_m": source_terminal,
    }
    return record, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--record-id", default="")
    args = parser.parse_args()
    if args.limit < 0 or args.shards < 1 or not 0 <= args.shard < args.shards or \
            (args.record_id and (args.limit or args.shards != 1)):
        raise ValueError("invalid shard or smoke selection")
    manifest = json.loads(args.manifest.read_text())
    preflight = json.loads(args.preflight.read_text())
    if manifest.get("schema") != "boundary_occupancy_rgb_replay_manifest_v1" or \
            manifest.get("group_size") != 4 or \
            manifest.get("seeds") != [11, 22, 33] or \
            digest(DATASET) != DATA_SHA or \
            manifest.get("source_sha256") != preflight.get("source_sha256") or \
            manifest.get("preflight_sha256") != digest(args.preflight) or \
            preflight.get("enough_coverage_for_rgb_replay") is not True:
        raise ValueError("invalid frozen boundary source")
    plans = [p for p in manifest["selected"][args.part]
             if shard_for(p, args.shards) == args.shard]
    if args.record_id:
        plans = [p for p in plans if p["record_id"] == args.record_id]
    plans = plans[:args.limit or None]
    if not plans or len({p["record_id"] for p in plans}) != len(plans):
        raise ValueError("empty or repeated source plans")
    infos = source_infos(manifest, args.root, plans)

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
    by_id = {str(episode.episode_id): episode for episode in dataset.episodes}
    if len(by_id) != len(dataset.episodes):
        raise ValueError("ambiguous train episode IDs")
    plans.sort(key=lambda p: (p["scene_id"], str(p["episode_id"]),
                              p["seed"], p["variant"]))
    requested = defaultdict(deque)
    selected = []
    for plan in plans:
        eid = str(plan["episode_id"])
        episode = by_id.get(eid)
        if episode is None or \
                canonical_scene(str(episode.scene_id)) != plan["scene_id"]:
            raise ValueError(f"missing train scene/episode: {eid}")
        selected.append(episode)
        requested[eid].append(plan)
    dataset.episodes = selected
    output = args.output_root / args.part
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = digest(args.manifest)
    completed, resumed, errors = 0, 0, []
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
                if record.get("manifest_sha256") == manifest_sha and \
                        audit.get("manifest_sha256") == manifest_sha and \
                        record.get("record_id") == audit.get("record_id") == rid and \
                        record.get("input", {}).get("instruction") == plan["instruction"] and \
                        record.get("input", {}).get("wrong_instruction") == plan["wrong_instruction"] and \
                        set(record["input"]["images"]) == {"outside", "inside"} and \
                        all((output / image).is_file()
                            for image in record["input"]["images"].values()):
                    completed += 1
                    resumed += 1
                    continue
            try:
                info = infos[(plan["seed"], eid, plan["variant"])]
                record, audit = collect_one(env, observation, plan, info,
                                            output, manifest_sha)
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
        "schema": "boundary_occupancy_rgb_collection_v1",
        "part": args.part, "manifest_sha256": manifest_sha,
        "requested": len(selected), "completed": completed,
        "unique_episode_ids": len({p["episode_id"] for p in plans}),
        "frames_expected": 2 * len(selected),
        "resumed": resumed, "errors": errors,
        "shard": args.shard, "shards": args.shards,
        "smoke_limit": args.limit, "target_record_id": args.record_id,
    }
    summary_name = "summary.json" if args.shards == 1 else \
                   f"summary.shard{args.shard}.json"
    write_json(output / summary_name, summary)
    if errors or completed != len(selected):
        raise RuntimeError(f"incomplete boundary RGB replay {completed}/{len(selected)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
