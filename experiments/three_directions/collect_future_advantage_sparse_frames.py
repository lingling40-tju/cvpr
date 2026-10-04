"""Replay only the initial and requested anchor RGB views for n=4 prefixes.

Model-input records and privileged geodesic replay audits are separate files.
Every executed action is replayed to check the terminal distance even though
RGB is saved only at turns 0, 3, and/or 6. This script needs a passed pooled
coverage gate and the manifest from prepare_future_advantage_sparse_manifest.
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


def group_shard(plan: dict, shards: int) -> int:
    key = f"{plan['seed']}:{plan['episode_id']}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % shards


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_infos(manifest: dict, root: Path,
                 plans: list[dict]) -> dict[tuple[int, str, int], dict]:
    wanted = {(plan["seed"], str(plan["episode_id"])) for plan in plans}
    result = {}
    for seed in sorted({seed for seed, _ in wanted}):
        rollout = root / f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
        audit = root / f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json"
        expected = manifest["source_sha256"]["seeds"][str(seed)]
        if digest(rollout) != expected["rollout"] or \
                digest(audit) != expected["train_audit"]:
            raise ValueError(f"source changed since sparse manifest: seed {seed}")
        groups = defaultdict(list)
        with rollout.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    eid = str(info["episode_id"])
                    if (seed, eid) in wanted:
                        groups[eid].append(info)
        for plan in (item for item in plans if item["seed"] == seed):
            eid, variant = str(plan["episode_id"]), plan["variant"]
            if len(groups[eid]) != 4:
                raise ValueError(f"source group is not n=4: {seed}/{eid}")
            info = groups[eid][variant]
            if info["end_reason"] != plan["terminal_mode"] or \
                    info["instruction"].strip() != plan["instruction"] or \
                    abs(float(info["distance_to_goal"]) -
                        plan["terminal_distance_m_for_replay_audit_only"]) > 1e-5:
                raise ValueError(f"source/manifest mismatch: {seed}/{eid}/{variant}")
            result[(seed, eid, variant)] = info
    if len(result) != len(plans):
        raise ValueError("source lookup coverage mismatch")
    return result


def collect_one(env, observation: dict, plan: dict, info: dict,
                output: Path, manifest_sha: str) -> tuple[dict, dict]:
    eid = str(plan["episode_id"])
    if str(env.current_episode.episode_id) != eid or \
            canonical_scene(str(env.current_episode.scene_id)) != plan["scene_id"] or \
            observation["instruction"]["text"].strip() != plan["instruction"]:
        raise ValueError(f"Habitat episode/scene/instruction mismatch: {eid}")
    rid = plan["record_id"]
    folder = output / "frames" / rid
    folder.mkdir(parents=True, exist_ok=True)
    initial = folder / "0000.jpg"
    save_frame(observation, initial)
    images = {"0": str(initial.relative_to(output))}
    requested = set(plan["anchor_turns"])
    if not requested or not requested.issubset({3, 6}):
        raise ValueError(f"invalid requested anchors: {rid}")
    start_distance = float(env.get_metrics()["distance_to_goal"])
    if not math.isfinite(start_distance):
        raise ValueError("nonfinite start distance")
    source_start = info.get("oracle_start_distance")
    if source_start is not None and abs(start_distance - float(source_start)) > .25:
        raise RuntimeError(f"start geodesic drift: {rid}")

    history = []
    audit_turns = []
    stop_seen = False
    before_distance = start_distance
    for turn_index, turn in enumerate(info["gen_traj"], 1):
        executed = list(turn.get("executed_actions", []))
        if not executed:
            if turn_index != len(info["gen_traj"]):
                raise ValueError(f"empty nonterminal turn: {rid}/{turn_index}")
            continue
        if stop_seen or "stop" in executed[:-1]:
            raise ValueError(f"action after STOP or early STOP: {rid}")
        motion = [action for action in executed if action != "stop"]
        for action in motion:
            code, repeat = action_steps(action)
            for _ in range(repeat):
                if env.episode_over:
                    raise RuntimeError(f"early replay termination: {rid}")
                observation = env.step({"action": code})
        stop_seen = "stop" in executed
        after_distance = float(env.get_metrics()["distance_to_goal"])
        if not math.isfinite(after_distance):
            raise ValueError("nonfinite replay distance")
        source_before = turn.get("oracle_before_distance")
        source_after = turn.get("oracle_after_distance")
        if ((source_before is not None and
             abs(before_distance - float(source_before)) > .25) or
            (source_after is not None and
             abs(after_distance - float(source_after)) > .25)):
            raise RuntimeError(f"turn geodesic drift: {rid}/{turn_index}")
        audit_turns.append({"turn": turn_index,
                            "before_distance_m": before_distance,
                            "after_distance_m": after_distance})
        before_distance = after_distance
        if motion:
            history.append({"turn": turn_index, "executed_actions": motion})
        if turn_index in requested:
            if not motion:
                raise ValueError(f"requested anchor has no motion: {rid}/{turn_index}")
            frame = folder / f"{turn_index:04d}.jpg"
            save_frame(observation, frame)
            images[str(turn_index)] = str(frame.relative_to(output))
    if set(map(int, images)) != {0} | requested:
        raise RuntimeError(f"missing sparse anchor RGB: {rid}")
    terminal_distance = float(env.get_metrics()["distance_to_goal"])
    source_terminal = float(plan["terminal_distance_m_for_replay_audit_only"])
    if abs(terminal_distance - source_terminal) > .25:
        raise RuntimeError(f"terminal replay drift: {rid}")
    model_input = {
        "schema": "future_advantage_sparse_model_input_v1",
        "manifest_sha256": manifest_sha, "record_id": rid,
        "input": {
            "instruction": plan["instruction"], "images": images,
            "action_history_by_anchor": {
                str(anchor): [turn for turn in history if turn["turn"] <= anchor]
                for anchor in sorted(requested)
            },
        },
    }
    replay_audit = {
        "schema": "future_advantage_sparse_replay_audit_v1",
        "manifest_sha256": manifest_sha, "record_id": rid,
        "seed": plan["seed"], "episode_id": eid,
        "variant": plan["variant"], "scene_id": plan["scene_id"],
        "start_distance_m": start_distance, "turns": audit_turns,
        "terminal_distance_m": terminal_distance,
        "source_terminal_distance_m": source_terminal,
        "terminal_drift_m": terminal_distance - source_terminal,
        "source_stop_response_omitted": stop_seen,
    }
    return model_input, replay_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard or smoke limit")
    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            manifest.get("group_size") != 4 or \
            manifest.get("anchors") != [3, 6] or \
            digest(DATASET) != manifest["source_sha256"]["dataset"] or \
            digest(args.report) != manifest.get("preflight_report_sha256") or \
            report.get("enough_coverage_for_fit_preparation") is not True or \
            report.get("seeds") != manifest.get("seeds") or \
            report.get("source_sha256") != manifest.get("source_sha256"):
        raise ValueError("invalid gated sparse replay manifest, report, or dataset")
    plans = [plan for plan in manifest["selected"][args.part]
             if group_shard(plan, args.shards) == args.shard]
    plans = plans[:args.limit or None]
    if not plans:
        raise ValueError("empty sparse replay selection")
    identities = [(p["seed"], str(p["episode_id"]), p["variant"]) for p in plans]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate selected trajectory")
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
            raise ValueError(f"manifest train episode missing: {eid}")
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
                record, audit = json.loads(record_path.read_text()), \
                    json.loads(audit_path.read_text())
                if record.get("manifest_sha256") == manifest_sha and \
                        audit.get("manifest_sha256") == manifest_sha and \
                        set(map(int, record["input"]["images"])) == \
                        {0} | set(plan["anchor_turns"]) and \
                        all((output / image).is_file()
                            for image in record["input"]["images"].values()):
                    completed += 1
                    resumed += 1
                    continue
            try:
                info = infos[(plan["seed"], eid, plan["variant"])]
                record, audit = collect_one(env, observation, plan, info,
                                            output, manifest_sha)
                atomic_json(audit_path, audit)
                atomic_json(record_path, record)
                completed += 1
                if completed % 50 == 0:
                    print(f"{args.part} {completed}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": rid, "error": repr(exc)})
                print(f"ERROR {rid}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete Habitat reset coverage")
    summary = {
        "schema": "future_advantage_sparse_collection_v1",
        "part": args.part, "manifest_sha256": manifest_sha,
        "requested": len(selected), "completed": completed,
        "unique_episode_ids": len({p["episode_id"] for p in plans}),
        "frames_expected": sum(1 + len(p["anchor_turns"]) for p in plans),
        "resumed": resumed, "errors": errors,
        "shard": args.shard, "shards": args.shards, "smoke_limit": args.limit,
    }
    summary_name = "summary.json" if args.shards == 1 else \
                   f"summary.shard{args.shard}.json"
    atomic_json(output / summary_name, summary)
    if errors or completed != len(selected):
        raise RuntimeError(f"incomplete sparse RGB replay {completed}/{len(selected)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
