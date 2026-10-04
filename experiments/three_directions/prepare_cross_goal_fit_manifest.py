"""Freeze train-only wrong goals before replaying any crossed-goal labels.

Every wrong instruction comes from another already frozen fit episode in the
same scene. Selection uses goal coordinates and instruction text only, never
policy outcomes or intermediate geodesic labels. Coordinates are stored for
simulator supervision and must not become reward-model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


IDS_SHA = "d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081"
RENDER_SHA = "43bf8bc8af46f07051d299810c1dd2975034a0b84da7f278b98fda5db761006c"
MIN_GOAL_SEPARATION_M = 3.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def goal(row: dict) -> list[float]:
    goals = row["goals"]
    if len(goals) != 1:
        raise ValueError("expected one R2R goal")
    position = [float(value) for value in (
        goals[0]["position"] if isinstance(goals[0], dict)
        else goals[0].position)]
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError("invalid R2R goal")
    return position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != IDS_SHA or digest(args.render_manifest) != RENDER_SHA:
        raise ValueError("frozen fit source changed")
    ids = json.loads(args.ids.read_text())
    render = json.loads(args.render_manifest.read_text())
    if (ids["schema"] != "control_exact512_policy_fit_extension_ids_v1"
            or ids["selected_episode_ids"] != 256
            or render["schema"] != "policy_process_train_manifest_v1"
            or render["source_id_manifest_sha256"] != IDS_SHA
            or render["train_dataset_sha256"] != digest(args.train_dataset)
            or render["targets"] != {"fit": 512, "development": 0, "audit": 0}):
        raise ValueError("fit manifest schema or identity mismatch")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        train = {str(row["episode_id"]): row
                 for row in json.load(stream)["episodes"]}
    fit_ids = {str(row["episode_id"]): row["scene_id"] for row in ids["rows"]}
    if len(fit_ids) != 256 or any(
            eid not in train or train[eid]["scene_id"] != scene
            for eid, scene in fit_ids.items()):
        raise ValueError("frozen fit ID or scene mismatch")
    by_scene = defaultdict(list)
    for eid, scene in fit_ids.items():
        by_scene[scene].append(train[eid])
    selected = {}
    ranked_candidates = {}
    skipped = []
    for eid in sorted(fit_ids):
        source = train[eid]
        candidates = []
        for other in by_scene[fit_ids[eid]]:
            oid = str(other["episode_id"])
            distance = math.dist(goal(source), goal(other))
            if (oid != eid and distance >= MIN_GOAL_SEPARATION_M
                    and source["instruction"]["instruction_text"].strip()
                    != other["instruction"]["instruction_text"].strip()):
                tie = hashlib.sha256(f"{eid}:{oid}".encode()).hexdigest()
                candidates.append((-distance, tie, other))
        if not candidates:
            skipped.append(eid)
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        ranked_candidates[eid] = [str(item[2]["episode_id"])
                                  for item in candidates]
        selected[eid] = candidates[0][2]
    plans = []
    for plan in render["selected"]["fit"]:
        eid = str(plan["episode_id"])
        if eid not in fit_ids or plan["scene_id"] != fit_ids[eid]:
            raise ValueError("render plan outside frozen fit")
        if eid not in selected:
            continue
        other = selected[eid]
        plans.append({"seed": int(plan["seed"]), "episode_id": eid,
                      "variant": int(plan["variant"]),
                      "scene_id": plan["scene_id"],
                      "wrong_episode_id": str(other["episode_id"]),
                      "wrong_goal_position_for_label_only": goal(other),
                      "wrong_instruction":
                          other["instruction"]["instruction_text"].strip()})
    counts = Counter(plan["episode_id"] for plan in plans)
    if (len(plans) != 2 * len(selected) or len(selected) < 200
            or set(counts.values()) != {2}
            or len({(p["episode_id"], p["variant"]) for p in plans})
            != len(plans)):
        raise ValueError("crossed-goal plan incomplete")
    report = {
        "schema": "cross_goal_fit_candidate_manifest_v1",
        "interpretation": "R2R-train fit-only goal coordinates for label replay; never policy input",
        "source_sha256": {"ids": IDS_SHA, "render_manifest": RENDER_SHA,
                          "train_dataset": digest(args.train_dataset)},
        "selection": "farthest Euclidean goal among other frozen fit IDs in the same scene, at least 3 m away, with distinct instruction; SHA tie break; before trajectory labels",
        "minimum_goal_separation_m": MIN_GOAL_SEPARATION_M,
        "episode_ids": len(selected), "trajectories": len(plans),
        "skipped_episode_ids": skipped,
        "ranked_candidate_episode_ids": ranked_candidates,
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"eligible_ids": len(selected), "plans": len(plans),
                      "skipped": len(skipped), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
