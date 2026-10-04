"""Replay frozen turn-3 audit routes into label-separated sparse RGB records."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

from collect_future_advantage_sparse_frames import (
    atomic_json, collect_one, group_shard, source_infos,
)
from collect_policy_preference_frames import CONFIG, DATASET, canonical_scene, digest


MANIFEST_SHA = "dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.limit < 0:
        raise ValueError("invalid audit shard or limit")
    manifest = json.loads(args.manifest.read_text())
    if digest(args.manifest) != MANIFEST_SHA or \
            manifest.get("schema") != "early_anchor_group4_audit_source_manifest_v1" or \
            manifest.get("group_size") != 4 or \
            manifest.get("anchor_turns") != [3] or \
            manifest.get("selected_records") != 540 or \
            digest(DATASET) != manifest["source_sha256"]["dataset"]:
        raise ValueError("frozen audit source changed")
    plans = [plan for plan in manifest["plans"]
             if group_shard(plan, args.shards) == args.shard]
    plans = plans[:args.limit or None]
    if not plans or len({p["record_id"] for p in plans}) != len(plans):
        raise ValueError("empty or duplicate audit shard")
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
    plans.sort(key=lambda plan: (plan["scene_id"], str(plan["episode_id"]),
                                 plan["seed"], plan["variant"]))
    selected, requested = [], defaultdict(deque)
    for plan in plans:
        eid = str(plan["episode_id"])
        episode = by_id.get(eid)
        if episode is None or \
                canonical_scene(str(episode.scene_id)) != plan["scene_id"]:
            raise ValueError(f"audit dataset episode missing: {eid}")
        selected.append(episode)
        requested[eid].append(plan)
    dataset.episodes = selected
    output = args.output_root / "audit"
    output.mkdir(parents=True, exist_ok=True)
    completed, resumed, errors = 0, 0, []
    with Env(config.TASK_CONFIG, dataset=dataset) as env:
        for _ in selected:
            observation = env.reset()
            eid = str(env.current_episode.episode_id)
            if not requested[eid]:
                raise RuntimeError(f"unexpected audit reset: {eid}")
            plan = requested[eid].popleft()
            rid = plan["record_id"]
            record_path = output / "records" / f"{rid}.json"
            audit_path = output / "audits" / f"{rid}.json"
            if record_path.is_file() and audit_path.is_file():
                record, audit = json.loads(record_path.read_text()), \
                    json.loads(audit_path.read_text())
                if record.get("manifest_sha256") == MANIFEST_SHA and \
                        audit.get("manifest_sha256") == MANIFEST_SHA and \
                        set(record["input"]["images"]) == {"0", "3"} and \
                        all((output / image).is_file()
                            for image in record["input"]["images"].values()):
                    completed += 1
                    resumed += 1
                    continue
            try:
                info = infos[(plan["seed"], eid, plan["variant"])]
                record, audit = collect_one(env, observation, plan, info,
                                            output, MANIFEST_SHA)
                atomic_json(audit_path, audit)
                atomic_json(record_path, record)
                completed += 1
                if completed % 50 == 0:
                    print(f"audit {completed}/{len(selected)}", flush=True)
            except Exception as exc:
                errors.append({"record_id": rid, "error": repr(exc)})
                print(f"ERROR {rid}: {exc!r}", flush=True)
    if any(requested.values()):
        raise RuntimeError("incomplete audit reset coverage")
    summary = {
        "schema": "early_anchor_group4_audit_collection_v1",
        "manifest_sha256": MANIFEST_SHA,
        "requested": len(selected), "completed": completed,
        "unique_episode_ids": len({p["episode_id"] for p in plans}),
        "frames_expected": 2 * len(selected), "resumed": resumed,
        "errors": errors, "shard": args.shard, "shards": args.shards,
        "smoke_limit": args.limit,
    }
    atomic_json(output / f"summary.shard{args.shard}.json", summary)
    if errors or completed != len(selected):
        raise RuntimeError(f"incomplete audit replay {completed}/{len(selected)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
