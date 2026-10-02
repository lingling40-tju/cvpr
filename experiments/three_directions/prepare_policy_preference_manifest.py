"""Freeze train-only successful/failed policy rollout pairs for visual audit.

Uses completed group-size-four seed 11 and 22 rollout logs. A failure must
remain farther than 3.5 m from the goal to avoid assigning opposite visual
labels to the same near-goal camera view when a policy merely omitted STOP.
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
    if not result or any(action not in VALID_ACTIONS for action in result):
        raise ValueError(f"invalid executed actions episode={info.get('episode_id')}")
    if "stop" in result[:-1]:
        raise ValueError("STOP before last executed action")
    return result


def chosen(info: dict, variant: int, seed: int) -> dict:
    return {"seed": seed, "variant": variant,
            "task_success": bool(info["task_success"]),
            "terminal_distance_m_for_replay_audit_only": float(info["distance_to_goal"]),
            "end_reason": info["end_reason"],
            "executed_actions": actions(info)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed11-rollout", type=Path, required=True)
    parser.add_argument("--seed22-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        train = {str(episode["episode_id"]): episode
                 for episode in json.load(stream)["episodes"]}
    records, per_seed = [], {}
    for seed, path in ((11, args.seed11_rollout), (22, args.seed22_rollout)):
        groups = defaultdict(list)
        steps = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(row["step"])
                for info in row["info"]:
                    eid = str(info["episode_id"])
                    if eid not in train or info["data_source"] != "r2r" or \
                            info["action_space"] != "r2r" or \
                            info["global_start_step"] != 1:
                        raise ValueError("unexpected rollout source")
                    if info["instruction"].strip() != \
                            train[eid]["instruction"]["instruction_text"].strip():
                        raise ValueError(f"instruction mismatch {eid}")
                    groups[eid].append(info)
        if steps != list(range(1, 129)) or len(groups) != 512 or \
                any(len(group) != 4 for group in groups.values()):
            raise ValueError(f"seed {seed}: incomplete 128-step group-four rollout")
        eligible = 0
        for eid, infos in groups.items():
            positives = [(i, info) for i, info in enumerate(infos)
                         if info["task_success"] and info["distance_to_goal"] <= 3.0]
            negatives = [(i, info) for i, info in enumerate(infos)
                         if not info["task_success"] and
                         math.isfinite(info["distance_to_goal"]) and
                         info["distance_to_goal"] >= 3.5]
            if not positives or not negatives:
                continue
            # Prefer a short success and the closest still unambiguously
            # unsuccessful endpoint as the hard visual negative.
            pi, pos = min(positives, key=lambda row: (len(actions(row[1])), row[0]))
            ni, neg = min(negatives,
                          key=lambda row: (row[1]["distance_to_goal"],
                                           len(actions(row[1])), row[0]))
            episode = train[eid]
            records.append({"episode_id": int(eid), "scene_id": episode["scene_id"],
                            "instruction": pos["instruction"],
                            "pair_id": f"seed{seed}_episode{eid}",
                            "success": chosen(pos, pi, seed),
                            "failure": chosen(neg, ni, seed)})
            eligible += 1
        per_seed[str(seed)] = {"groups": len(groups), "eligible_pairs": eligible,
                              "rollout_path": str(path), "rollout_sha256": digest(path)}
    scenes = sorted({row["scene_id"] for row in records},
                    key=lambda scene: hashlib.sha256(
                        ("policy-preference-v1:" + scene).encode()).hexdigest())
    if len(scenes) < 24:
        raise ValueError("too few eligible train scenes")
    development, audit = set(scenes[:8]), set(scenes[8:16])
    split = lambda scene: ("development" if scene in development else
                           "audit" if scene in audit else "fit")
    for row in records:
        row["split"] = split(row["scene_id"])
    counts = Counter(row["split"] for row in records)
    if min(counts.values()) < 20:
        raise ValueError(f"underpowered held-out split: {counts}")
    payload = {"schema": "policy_preference_v1", "train_sha256": digest(DATASET),
               "selection": {"group_size": 4, "optimizer_steps_per_seed": 128,
                             "failed_distance_at_least_m": 3.5,
                             "fit_scenes": len(scenes) - 16,
                             "development_scenes": 8, "audit_scenes": 8},
               "sources": per_seed,
               "scene_split": {"fit": sorted(set(scenes) - development - audit),
                               "development": sorted(development),
                               "audit": sorted(audit)},
               "counts": dict(counts),
               "pairs": sorted(records, key=lambda row: row["pair_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pairs": len(records), "scenes": len(scenes),
                      "counts": counts, "per_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()
