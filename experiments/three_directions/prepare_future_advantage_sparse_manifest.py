"""Freeze sparse RGB replay inputs for an audited n=4 future-advantage fit.

Only fit/development train scenes are selected. Pair labels are written to a
separate file so that replay and model-input code need not read the privileged
future geodesic return. The pooled coverage gate must have passed first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path

from preflight_group_future_advantage import digest
from preflight_group_future_advantage_pool import coverage_checks


ANCHORS = (3, 6)
STEPS = 128
GROUPS_PER_STEP = 4
ROLLOUTS_PER_GROUP = 4


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build(report: dict, root: Path, dataset: Path, scene_split: Path,
          *, require_gate: bool = True) -> tuple[dict, dict]:
    if report.get("schema") != "group_future_advantage_exact512_pooled_preflight_v1" or \
            report.get("group_size") != ROLLOUTS_PER_GROUP or \
            report.get("steps_per_seed") != STEPS or \
            report.get("unique_train_episodes") != STEPS * GROUPS_PER_STEP:
        raise ValueError("not the audited exact512 n=4 pooled inventory")
    checks = coverage_checks(report)
    if report.get("coverage_checks") != checks or \
            report.get("enough_coverage_for_fit_preparation") is not all(
                checks.values()):
        raise ValueError("pooled coverage report decision is inconsistent")
    if require_gate and report.get("enough_coverage_for_fit_preparation") is not True:
        raise ValueError("frozen coverage gate failed: no RGB replay")
    seeds = report.get("seeds")
    if seeds not in ([11, 22], [11, 22, 33]):
        raise ValueError("unexpected staged seed selection")
    sources = report["source_sha256"]
    if digest(dataset) != sources["dataset"] or \
            digest(scene_split) != sources["scene_split"]:
        raise ValueError("frozen train dataset or scene split changed")
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(stream)["episodes"]}
    split = json.loads(scene_split.read_text())["scene_split"]
    scene_to_part = {scene: part for part, scenes in split.items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split.values())) or \
            not {"fit", "development"}.issubset(split):
        raise ValueError("invalid scene-disjoint split")

    selected: dict[str, dict[str, dict]] = {"fit": {}, "development": {}}
    pair_labels: dict[str, list[dict]] = {"fit": [], "development": []}
    first_order = None
    for seed in seeds:
        rollout = root / f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
        audit = root / f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json"
        expected = sources["seeds"][str(seed)]
        if digest(rollout) != expected["rollout"] or \
                digest(audit) != expected["train_audit"]:
            raise ValueError(f"audited source changed: seed {seed}")
        audit_data = json.loads(audit.read_text())
        if audit_data.get("seed") != seed or audit_data.get("steps") != STEPS or \
                audit_data.get("group_size") != 4 or \
                audit_data.get("same_row_candidate_control") is not True or \
                audit_data.get("nonzero_actor_gradient_steps", {}).get(
                    "candidate") != STEPS:
            raise ValueError(f"invalid n=4 training audit: seed {seed}")
        seen = set()
        order = []
        with rollout.open() as stream:
            for step, line in enumerate(stream, 1):
                batch = json.loads(line)
                if step > STEPS or batch.get("step") != step or \
                        len(batch.get("info", [])) != 16:
                    raise ValueError(f"invalid rollout step {step}: seed {seed}")
                groups = defaultdict(list)
                for info in batch["info"]:
                    groups[str(info["episode_id"])].append(info)
                if len(groups) != GROUPS_PER_STEP or any(
                        len(group) != ROLLOUTS_PER_GROUP
                        for group in groups.values()):
                    raise ValueError(f"not four same-episode rollouts: seed {seed}")
                if seen.intersection(groups):
                    raise ValueError(f"repeated episode within seed {seed}")
                seen.update(groups)
                order.append(sorted(groups))
                for eid, infos in groups.items():
                    episode = episodes.get(eid)
                    if episode is None:
                        raise ValueError(f"unknown train episode {eid}")
                    scene = str(episode["scene_id"])
                    part = scene_to_part.get(scene)
                    if part is None:
                        raise ValueError(f"scene absent from frozen split: {scene}")
                    if part not in selected or any(bool(x["task_success"])
                                                    for x in infos):
                        continue
                    for anchor in ANCHORS:
                        active = [(variant, info) for variant, info in enumerate(infos)
                                  if len(info["gen_traj"]) > anchor]
                        for (left_id, left), (right_id, right) in \
                                itertools.combinations(active, 2):
                            left_future = sum(float(turn["oracle_turn_progress"])
                                              for turn in left["gen_traj"][anchor:])
                            right_future = sum(float(turn["oracle_turn_progress"])
                                               for turn in right["gen_traj"][anchor:])
                            gap = left_future - right_future
                            if not math.isfinite(gap):
                                raise ValueError("nonfinite future return gap")
                            if abs(gap) < .25:
                                continue
                            pair_labels[part].append({
                                "pair_id": f"s{seed}_e{eid}_a{anchor}_v{left_id}_{right_id}",
                                "seed": seed, "episode_id": eid, "scene_id": scene,
                                "anchor": anchor, "left_variant": left_id,
                                "right_variant": right_id,
                                "preferred_variant": left_id if gap > 0 else right_id,
                                "future_return_gap_for_label_only": gap,
                                "same_terminal_mode":
                                    left["end_reason"] == right["end_reason"],
                            })
                            for variant, info in ((left_id, left), (right_id, right)):
                                rid = f"s{seed}_e{eid}_v{variant}"
                                item = selected[part].setdefault(rid, {
                                    "record_id": rid, "seed": seed,
                                    "episode_id": eid, "variant": variant,
                                    "scene_id": scene, "instruction":
                                        info["instruction"].strip(),
                                    "terminal_mode": info["end_reason"],
                                    "terminal_distance_m_for_replay_audit_only":
                                        float(info["distance_to_goal"]),
                                    "anchor_turns": [],
                                })
                                if anchor not in item["anchor_turns"]:
                                    item["anchor_turns"].append(anchor)
        if len(order) != STEPS or len(seen) != STEPS * GROUPS_PER_STEP:
            raise ValueError(f"incomplete seed {seed} rollout")
        if first_order is None:
            first_order = order
        elif order != first_order:
            raise ValueError("train episode rows differ across seeds")

    counts = {}
    for part in selected:
        counts[part] = {}
        for anchor in ANCHORS:
            pairs = [pair for pair in pair_labels[part]
                     if pair["anchor"] == anchor]
            groups = {pair["episode_id"] for pair in pairs}
            same_mode = [pair for pair in pairs if pair["same_terminal_mode"]]
            same_groups = {pair["episode_id"] for pair in same_mode}
            expected = report["parts"][part]
            broad = expected["anchors"][str(anchor)]
            same = expected["same_terminal_mode_anchors"][str(anchor)]
            if (len(pairs), len(groups), len(same_mode), len(same_groups)) != \
                    (broad["pairs"], broad["episode_groups"],
                     same["pairs"], same["episode_groups"]):
                raise ValueError(f"manifest/inventory mismatch: {part}/{anchor}")
            counts[part][str(anchor)] = {
                "pairs": len(pairs), "unique_episode_groups": len(groups),
                "same_terminal_mode_pairs": len(same_mode),
                "same_terminal_mode_unique_episode_groups": len(same_groups),
            }
    replay = {
        "schema": "future_advantage_sparse_replay_manifest_v1",
        "seeds": seeds, "anchors": list(ANCHORS), "group_size": 4,
        "source_sha256": sources,
        "selected": {part: sorted(items.values(),
                            key=lambda row: (row["scene_id"], row["episode_id"],
                                             row["seed"], row["variant"]))
                     for part, items in selected.items()},
        "counts": counts,
        "interpretation": "Train-scene RGB replay selection; no future labels are model inputs.",
    }
    for part in replay["selected"]:
        for item in replay["selected"][part]:
            item["anchor_turns"].sort()
    labels = {
        "schema": "future_advantage_within_group_pair_labels_v1",
        "seeds": seeds, "group_size": 4, "anchors": list(ANCHORS),
        "pairs": {part: sorted(items, key=lambda row: row["pair_id"])
                  for part, items in pair_labels.items()},
        "counts": counts,
        "interpretation": "Privileged future geodesic return is training/development supervision only.",
    }
    return replay, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_output == args.labels_output:
        raise ValueError("replay and label outputs must be separate")
    replay, labels = build(json.loads(args.report.read_text()), args.root,
                           args.dataset, args.scene_split)
    replay["preflight_report_sha256"] = digest(args.report)
    atomic_json(args.replay_output, replay)
    labels["replay_manifest_sha256"] = digest(args.replay_output)
    labels["preflight_report_sha256"] = digest(args.report)
    atomic_json(args.labels_output, labels)
    print(json.dumps({"seeds": replay["seeds"], "counts": replay["counts"],
                      "selected": {part: len(rows) for part, rows in
                                   replay["selected"].items()}}, indent=2))


if __name__ == "__main__":
    main()
