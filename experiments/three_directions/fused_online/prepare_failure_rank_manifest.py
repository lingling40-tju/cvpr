"""Select same-instruction failed trajectory pairs for reward pretraining.

Only train-split rollout logs and train-split simulator distances are used.
The distance values select and later audit examples; online reward inference
must use images and the instruction without access to simulator distance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DATASET = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
VALID_ACTIONS = {"stop"} | {f"move forward {n}cm" for n in (25, 50, 75)} | \
    {f"turn {direction} {n} degrees" for direction in ("left", "right")
     for n in (15, 30, 45)}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def actions(info: dict) -> list[str]:
    result = [str(action) for turn in info.get("gen_traj", [])
              for action in turn.get("executed_actions", [])]
    if len([x for x in result if x != "stop"]) < 3 or \
            any(action not in VALID_ACTIONS for action in result) or \
            "stop" in result[:-1]:
        raise ValueError(f"invalid executed actions episode={info.get('episode_id')}")
    return result


def option(info: dict, variant: int, seed: int) -> dict:
    executed = actions(info)
    boundaries = []
    count = 0
    for turn in info["gen_traj"]:
        count += sum(action != "stop" for action in turn.get("executed_actions", []))
        boundaries.append(count)
    if not boundaries or count != sum(action != "stop" for action in executed):
        raise ValueError("turn/action boundary mismatch")
    return {"seed": seed, "variant": variant, "task_success": False,
            "terminal_distance_m_for_replay_audit_only": float(info["distance_to_goal"]),
            "end_reason": info["end_reason"], "executed_actions": executed,
            "turn_action_boundaries": boundaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        train = {str(episode["episode_id"]): episode
                 for episode in json.load(stream)["episodes"]}
    records, sources = [], {}
    for seed in (11, 22, 33):
        experiment = "three_directions_group4_128step"
        if seed != 11:
            experiment += f"_seed{seed}"
        run = args.source_root / "runlogs" / experiment
        path = args.source_root / "verl_checkpoints" / experiment / "rollout.jsonl"
        if not (run / "completed").is_file():
            raise ValueError(f"incomplete source seed {seed}")
        groups = defaultdict(list)
        steps = []
        for line in path.open():
            row = json.loads(line)
            steps.append(row["step"])
            for info in row["info"]:
                eid = str(info["episode_id"])
                if eid not in train or info["data_source"] != "r2r" or \
                        info["action_space"] != "r2r" or info["global_start_step"] != 1:
                    raise ValueError("unexpected training rollout source")
                if info["instruction"].strip() != \
                        train[eid]["instruction"]["instruction_text"].strip():
                    raise ValueError(f"instruction mismatch episode {eid}")
                groups[eid].append(info)
        if steps != list(range(1, 129)) or len(groups) != 512 or \
                any(len(group) != 4 for group in groups.values()):
            raise ValueError(f"incomplete group-four rollout seed {seed}")
        selected = 0
        for eid, infos in groups.items():
            eligible = []
            for index, info in enumerate(infos):
                distance = float(info["distance_to_goal"])
                if info["task_success"] or not math.isfinite(distance) or distance < 3.5:
                    continue
                try:
                    motion = actions(info)
                except ValueError:
                    continue
                eligible.append((distance, len(motion), index, info))
            if len(eligible) < 2:
                continue
            near = min(eligible)
            far = max(eligible, key=lambda x: (x[0], -x[1], -x[2]))
            if far[0] - near[0] < 1.5:
                continue
            scene_id = train[eid]["scene_id"]
            records.append({"episode_id": int(eid), "scene_id": scene_id,
                            "instruction": near[3]["instruction"],
                            "pair_id": f"seed{seed}_episode{eid}",
                            "distance_gap_m_for_selection_only": far[0] - near[0],
                            "near": option(near[3], near[2], seed),
                            "far": option(far[3], far[2], seed)})
            selected += 1
        sources[str(seed)] = {"groups": len(groups), "selected_pairs": selected,
                              "rollout_path": str(path), "rollout_sha256": digest(path)}
    scenes = sorted({row["scene_id"] for row in records},
                    key=lambda scene: hashlib.sha256(
                        ("failure-rank-v1:" + scene).encode()).hexdigest())
    if len(scenes) < 24:
        raise ValueError("too few train scenes for scene-disjoint splits")
    development, audit = set(scenes[:8]), set(scenes[8:16])
    for row in records:
        row["split"] = ("development" if row["scene_id"] in development else
                        "audit" if row["scene_id"] in audit else "fit")
    counts = Counter(row["split"] for row in records)
    if min(counts.values()) < 25:
        raise ValueError(f"underpowered scene split: {counts}")
    manifest = {"schema": "failure_rank_train_scene_v1",
                "interpretation": "Train-split, scene-disjoint ordinal reward data; no val-unseen or RL result.",
                "train_sha256": digest(DATASET),
                "selection": {"group_size": 4, "optimizer_steps_per_seed": 128,
                              "failed_distance_at_least_m": 3.5,
                              "near_far_gap_at_least_m": 1.5,
                              "fit_scenes": len(scenes) - 16,
                              "development_scenes": 8, "audit_scenes": 8},
                "sources": sources,
                "scene_split": {"fit": sorted(set(scenes) - development - audit),
                                "development": sorted(development),
                                "audit": sorted(audit)},
                "counts": dict(counts),
                "pairs": sorted(records, key=lambda row: row["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pairs": len(records), "scenes": len(scenes),
                      "counts": counts, "sources": sources,
                      "manifest_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
