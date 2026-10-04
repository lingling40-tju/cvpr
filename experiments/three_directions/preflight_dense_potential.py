"""Train-only coverage check for a dense visual potential candidate.

The input is the frozen n=4 sparse future-advantage source manifest. This
reads simulator distances only to count possible *training* labels; it never
renders images, fits a model, or reads val-unseen trajectories.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


THRESHOLDS = (0.1, 0.25, 0.5, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records(root: Path, manifest: dict, plans: list[dict]) -> dict:
    wanted = defaultdict(set)
    for plan in plans:
        wanted[plan["seed"]].add(str(plan["episode_id"]))
    result = {}
    for seed, episode_ids in sorted(wanted.items()):
        rollout = root / (
            f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
        )
        audit = root / f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json"
        expected = manifest["source_sha256"]["seeds"][str(seed)]
        if digest(rollout) != expected["rollout"] or \
                digest(audit) != expected["train_audit"]:
            raise ValueError(f"frozen training source changed: seed {seed}")
        groups = defaultdict(list)
        with rollout.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    eid = str(info["episode_id"])
                    if eid in episode_ids:
                        groups[eid].append(info)
        for plan in (p for p in plans if p["seed"] == seed):
            eid, variant = str(plan["episode_id"]), int(plan["variant"])
            if len(groups[eid]) != 4:
                raise ValueError(f"source group is not n=4: {seed}/{eid}")
            info = groups[eid][variant]
            if info["instruction"].strip() != plan["instruction"] or \
                    info["end_reason"] != plan["terminal_mode"] or \
                    abs(float(info["distance_to_goal"]) -
                        float(plan["terminal_distance_m_for_replay_audit_only"])) > 1e-5:
                raise ValueError(f"source/manifest mismatch: {seed}/{eid}/{variant}")
            result[plan["record_id"]] = info
    if len(result) != len(plans):
        raise ValueError("source record coverage mismatch")
    return result


def analyze_part(plans: list[dict], records: dict) -> dict:
    by_window = {name: Counter() for name in ("turns_1_3", "turns_4_6", "turns_7_plus", "all")}
    episode_with_forward = {threshold: set() for threshold in THRESHOLDS}
    episode_with_backward = {threshold: set() for threshold in THRESHOLDS}
    episode_with_ordered_forward = set()
    episode_with_ordered_backward = set()
    ordered = Counter()
    anchor_pairs = {3: Counter(), 6: Counter()}
    anchor_pair_groups = {3: set(), 6: set()}
    by_group = defaultdict(list)
    turn_counts = Counter()
    scene_ids = set()
    episode_ids = set()
    for plan in plans:
        info = records[plan["record_id"]]
        scene_ids.add(plan["scene_id"])
        episode_key = (int(plan["seed"]), str(plan["episode_id"]))
        episode_ids.add(episode_key)
        by_group[episode_key].append((plan, info))
        trajectory = info["gen_traj"]
        if not trajectory:
            raise ValueError(f"empty trajectory: {plan['record_id']}")
        distances = [float(trajectory[0]["oracle_before_distance"])]
        if not all(math.isfinite(value) for value in distances):
            raise ValueError("nonfinite start distance")
        for index, turn in enumerate(trajectory, 1):
            before = float(turn["oracle_before_distance"])
            after = float(turn["oracle_after_distance"])
            if not math.isfinite(before) or not math.isfinite(after) or \
                    abs(before - distances[-1]) > 1e-4:
                raise ValueError(f"discontinuous geodesic trace: {plan['record_id']}/{index}")
            distances.append(after)
            if not turn.get("executed_actions"):
                continue
            window = "turns_1_3" if index <= 3 else (
                "turns_4_6" if index <= 6 else "turns_7_plus")
            delta = before - after
            turn_counts[window] += 1
            turn_counts["all"] += 1
            for name in (window, "all"):
                for threshold in THRESHOLDS:
                    key = f"{threshold:g}"
                    if delta >= threshold:
                        by_window[name][f"forward_ge_{key}m"] += 1
                    if delta <= -threshold:
                        by_window[name][f"backward_ge_{key}m"] += 1
            for threshold in THRESHOLDS:
                if delta >= threshold:
                    episode_with_forward[threshold].add(episode_key)
                if delta <= -threshold:
                    episode_with_backward[threshold].add(episode_key)
        for earlier, before in enumerate(distances):
            for later in range(earlier + 1, len(distances)):
                gap = before - distances[later]
                if gap >= 1.0:
                    ordered["forward_ge_1m"] += 1
                    episode_with_ordered_forward.add(episode_key)
                elif gap <= -1.0:
                    ordered["backward_ge_1m"] += 1
                    episode_with_ordered_backward.add(episode_key)
    for episode_key, group in by_group.items():
        for anchor in (3, 6):
            eligible = [(plan, info) for plan, info in group
                        if anchor in plan["anchor_turns"] and
                        len(info["gen_traj"]) >= anchor]
            for index, (left_plan, left) in enumerate(eligible):
                for right_plan, right in eligible[index + 1:]:
                    left_distance = float(left["gen_traj"][anchor - 1][
                        "oracle_after_distance"])
                    right_distance = float(right["gen_traj"][anchor - 1][
                        "oracle_after_distance"])
                    if abs(left_distance - right_distance) < 1.0:
                        continue
                    anchor_pairs[anchor]["distance_gap_ge_1m"] += 1
                    anchor_pair_groups[anchor].add(episode_key)
                    if left_plan["terminal_mode"] == right_plan["terminal_mode"]:
                        anchor_pairs[anchor]["same_terminal_mode"] += 1
    return {
        "records": len(plans),
        "unique_seed_episode_groups": len(episode_ids),
        "scenes": len(scene_ids),
        "executed_turns": dict(turn_counts),
        "turn_labels": {name: dict(counts) for name, counts in by_window.items()},
        "episode_groups_with_both_turn_directions": {
            f"{threshold:g}m": len(episode_with_forward[threshold] &
                                     episode_with_backward[threshold])
            for threshold in THRESHOLDS
        },
        "ordered_state_pairs": dict(ordered),
        "episode_groups_with_both_ordered_state_directions": len(
            episode_with_ordered_forward & episode_with_ordered_backward),
        "same_episode_anchor_pairs": {
            str(anchor): {**dict(anchor_pairs[anchor]),
                          "unique_seed_episode_groups": len(anchor_pair_groups[anchor])}
            for anchor in (3, 6)
        },
    }


def goal_swap_coverage(plans: list[dict], episodes: list[dict]) -> dict:
    selected_ids = {str(plan["episode_id"]) for plan in plans}
    scenes = {plan["scene_id"] for plan in plans}
    by_start = defaultdict(list)
    for episode in episodes:
        scene = str(episode["scene_id"])
        if scene.startswith("data/scene_datasets/"):
            scene = scene[len("data/scene_datasets/"):]
        if scene not in scenes:
            continue
        key = (scene,
               tuple(round(float(x), 4) for x in episode["start_position"]),
               tuple(round(float(x), 4) for x in episode["start_rotation"]))
        goal = tuple(float(x) for x in episode["goals"][0]["position"])
        by_start[key].append((str(episode["episode_id"]), goal))
    selected_with_alternative = set()
    selected_both_ids = set()
    both_selected_pairs = 0
    starts_with_selected_pair = 0
    for rows in by_start.values():
        start_has_pair = False
        for index, (left_id, left_goal) in enumerate(rows):
            for right_id, right_goal in rows[index + 1:]:
                if math.dist(left_goal, right_goal) < 1.0:
                    continue
                if left_id in selected_ids:
                    selected_with_alternative.add(left_id)
                if right_id in selected_ids:
                    selected_with_alternative.add(right_id)
                if left_id in selected_ids and right_id in selected_ids:
                    both_selected_pairs += 1
                    start_has_pair = True
                    selected_both_ids.update((left_id, right_id))
        starts_with_selected_pair += int(start_has_pair)
    return {
        "selected_unique_episode_ids": len(selected_ids),
        "selected_ids_with_same_start_alternative_goal_in_dataset": len(
            selected_with_alternative),
        "same_start_different_goal_pairs_with_both_ids_selected": both_selected_pairs,
        "unique_selected_ids_in_both_selected_pairs": len(selected_both_ids),
        "starts_with_both_selected_goal_ids": starts_with_selected_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            manifest.get("group_size") != 4 or manifest.get("seeds") != [11, 22]:
        raise ValueError("unexpected frozen group-four source")
    if digest(args.dataset) != manifest["source_sha256"]["dataset"]:
        raise ValueError("train dataset changed")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    selected = manifest["selected"]
    if set(selected) != {"fit", "development"}:
        raise ValueError("unexpected scene partitions")
    plans = selected["fit"] + selected["development"]
    if len({p["record_id"] for p in plans}) != len(plans):
        raise ValueError("duplicate source record")
    records = source_records(args.root, manifest, plans)
    result = {
        "schema": "dense_instruction_potential_train_source_preflight_v1",
        "group_size": 4,
        "source_manifest_sha256": digest(args.manifest),
        "train_dataset_sha256": digest(args.dataset),
        "thresholds_m": list(THRESHOLDS),
        "parts": {
            name: {**analyze_part(selected[name], records),
                   "goal_swap_metadata": goal_swap_coverage(selected[name], episodes)}
            for name in ("fit", "development")
        },
        "interpretation": "Train-only label coverage. No new RGB, learned potential, reward, or navigation result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
