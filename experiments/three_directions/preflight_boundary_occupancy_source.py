"""Freeze train-scene RGB boundary source before viewing any images.

Labels come from simulator distance in audited n=4 policy rollouts. They are
source-selection labels only, never inputs to an observation-only model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


DATA_SHA = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
SOURCE_SHA = {
    11: ("944d8ab37f48eef9d83ebfaff31b1ea52dddfdf991e35b6dacb61a46ac94203d",
         "4ef819c3682eb831369850e4ef8ef3b90716827d316d8c5a260add78760b5983"),
    22: ("521b434159de1b6b6e9b86773acc5cfe045563d91a56941c706c80e2b8177512",
         "d1db7dc856c66c97cb28265d67b791df26244b4fe712b55bfd99911ac1e92364"),
    33: ("e033f264673c6ab194918fbce66bd2fb133a09fdf81f87dbeb961d718b08a4be",
         "cdbec2e70ade62b0b23f71af8d54d48d28d1b894ab870671425a75bbe75ba6b3"),
}
SPLIT_SALT = "boundary-occupancy-source-v1:"
DEVELOPMENT_SCENES = 8
AUDIT_SCENES = 8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_groups(path: Path):
    seen = set()
    for step, line in enumerate(path.open(), 1):
        batch = json.loads(line)
        if batch.get("step") != step or len(batch.get("info", [])) != 16:
            raise ValueError(f"invalid n4 step {step}: {path}")
        groups = defaultdict(list)
        for info in batch["info"]:
            if info.get("data_source") != "r2r":
                raise ValueError("non-R2R training row")
            groups[str(info["episode_id"])].append(info)
        if len(groups) != 4 or any(len(v) != 4 for v in groups.values()) or \
                seen.intersection(groups):
            raise ValueError(f"duplicate or non-n4 group: {path}:{step}")
        seen.update(groups)
        yield step, groups
    if step != 128 or len(seen) != 512:
        raise ValueError(f"incomplete exact512 rollout: {path}")


def boundary_pair(distances: list[float]) -> tuple[int, int] | None:
    """First just-outside to inside pair within at most two turns."""
    for before in range(len(distances) - 1):
        if 3.5 <= distances[before] <= 4.5:
            for after in range(before + 1, min(before + 3, len(distances))):
                if distances[after] <= 3.0:
                    return before, after
    return None


def point_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dataset) != DATA_SHA:
        raise ValueError("R2R-train dataset changed")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episode_rows = json.load(stream)["episodes"]
    episodes = {str(row["episode_id"]): row for row in episode_rows}
    if len(episodes) != len(episode_rows):
        raise ValueError("duplicate train dataset episode")
    paths = {}
    first_order = None
    source_scenes = set()
    for seed, (expected_rollout, expected_audit) in SOURCE_SHA.items():
        rollout = args.root / f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
        audit = args.root / f"runlogs/oracle_exact512_scale/train_audit_seed{seed}.json"
        if digest(rollout) != expected_rollout or digest(audit) != expected_audit:
            raise ValueError(f"seed {seed} source changed")
        audit_value = json.loads(audit.read_text())
        if audit_value.get("seed") != seed or audit_value.get("steps") != 128 or \
                audit_value.get("group_size") != 4 or \
                audit_value.get("same_row_candidate_control") is not True:
            raise ValueError(f"seed {seed} train audit mismatch")
        paths[seed] = rollout
        order = []
        for _, groups in ordered_groups(rollout):
            order.append(sorted(groups))
            for eid in groups:
                if eid not in episodes:
                    raise ValueError(f"unknown R2R-train episode {eid}")
                source_scenes.add(str(episodes[eid]["scene_id"]))
        if first_order is None:
            first_order = order
        elif order != first_order:
            raise ValueError("different train episode rows across seeds")

    # Scene assignment sees only IDs and scene names, never rewards or distances.
    scene_order = sorted(source_scenes, key=lambda scene:
                         hashlib.sha256((SPLIT_SALT + scene).encode()).hexdigest())
    if len(scene_order) < 3 * DEVELOPMENT_SCENES:
        raise ValueError("too few training scenes for a frozen holdout")
    split = {
        "development": scene_order[:DEVELOPMENT_SCENES],
        "audit": scene_order[DEVELOPMENT_SCENES:
                             DEVELOPMENT_SCENES + AUDIT_SCENES],
        "fit": scene_order[DEVELOPMENT_SCENES + AUDIT_SCENES:],
    }
    scene_to_part = {scene: part for part, scenes in split.items()
                     for scene in scenes}
    scene_episodes = defaultdict(list)
    for item in episode_rows:
        scene_episodes[str(item["scene_id"])].append(item)

    records = {part: [] for part in split}
    counts = {part: Counter() for part in split}
    ids = {part: defaultdict(set) for part in split}
    per_seed = {seed: Counter() for seed in SOURCE_SHA}
    for seed, rollout in paths.items():
        for _, groups in ordered_groups(rollout):
            for eid, four in groups.items():
                episode = episodes[eid]
                scene = str(episode["scene_id"])
                part = scene_to_part[scene]
                counts[part]["episode_groups_per_seed"] += 1
                for variant, info in enumerate(four):
                    start = float(info["oracle_start_distance"])
                    if not math.isfinite(start) or start < 0:
                        raise ValueError("invalid start geodesic")
                    distances = [start]
                    for turn in info["gen_traj"]:
                        before = float(turn["oracle_before_distance"])
                        after = float(turn["oracle_after_distance"])
                        if not all(math.isfinite(x) and x >= 0 for x in
                                   (before, after)) or \
                                abs(before - distances[-1]) > 1e-4:
                            raise ValueError("discontinuous turnwise geodesic")
                        distances.append(after)
                    if not info["gen_traj"] or abs(
                            distances[-1] - float(info["distance_to_goal"])) > 1e-4:
                        raise ValueError("terminal replay distance mismatch")
                    success = bool(info["task_success"])
                    inside = min(distances) <= 3.0
                    pair = boundary_pair(distances)
                    counts[part]["rollouts"] += 1
                    counts[part]["ever_inside_3m"] += inside
                    counts[part]["failed_ever_inside_3m"] += inside and not success
                    if inside:
                        ids[part]["ever_inside_3m"].add(eid)
                    if inside and not success:
                        ids[part]["failed_ever_inside_3m"].add(eid)
                    if pair is None:
                        continue
                    before, after = pair
                    counts[part]["paired_boundary_rollouts"] += 1
                    per_seed[seed]["paired_boundary_rollouts"] += 1
                    ids[part]["paired_boundary"].add(eid)
                    if not success:
                        ids[part]["paired_boundary_failure"].add(eid)
                    true_goal = episode["goals"][0]["position"]
                    true_start = episode["start_position"]
                    alternatives = [other for other in scene_episodes[scene]
                                    if str(other["episode_id"]) != eid and
                                    point_distance(true_goal,
                                                   other["goals"][0]["position"]) > 7.0]
                    exact = [other for other in alternatives if
                             point_distance(true_start, other["start_position"]) < 1e-4]
                    counts[part]["paired_with_far_wrong_instruction"] += bool(alternatives)
                    counts[part]["paired_with_exact_start_far_wrong_instruction"] += bool(exact)
                    if alternatives:
                        ids[part]["far_wrong_instruction"].add(eid)
                    if exact:
                        ids[part]["exact_start_far_wrong_instruction"].add(eid)
                    records[part].append({
                        "record_id": f"s{seed}_e{eid}_v{variant}",
                        "seed": seed, "episode_id": eid, "scene_id": scene,
                        "variant": variant,
                        "outside_state_index": before,
                        "inside_state_index": after,
                        "outside_distance_m_for_label_only": distances[before],
                        "inside_distance_m_for_label_only": distances[after],
                        "task_success_for_audit_only": success,
                        "far_wrong_instruction_episode_id":
                            str(alternatives[0]["episode_id"]) if alternatives else None,
                        "exact_start_far_wrong_instruction_episode_id":
                            str(exact[0]["episode_id"]) if exact else None,
                    })
    summaries = {}
    for part in split:
        summary = dict(counts[part])
        summary["scenes"] = len(split[part])
        summary["unique_episode_ids"] = {
            key: len(value) for key, value in ids[part].items()}
        summaries[part] = summary
    gates = {
        "fit_at_least_150_paired_episode_ids":
            len(ids["fit"]["paired_boundary"]) >= 150,
        "development_at_least_50_paired_episode_ids":
            len(ids["development"]["paired_boundary"]) >= 50,
        "audit_at_least_50_paired_episode_ids":
            len(ids["audit"]["paired_boundary"]) >= 50,
        "development_at_least_30_far_wrong_instruction_ids":
            len(ids["development"]["far_wrong_instruction"]) >= 30,
    }
    result = {
        "schema": "boundary_occupancy_train_scene_source_preflight_v1",
        "source_sha256": {"dataset": DATA_SHA,
                          "seeds": {str(seed): {"rollout": pair[0], "train_audit": pair[1]}
                                    for seed, pair in SOURCE_SHA.items()}},
        "group_size": 4, "seeds": list(SOURCE_SHA),
        "split_rule": f"SHA-256({SPLIT_SALT}+scene); first 8 development, next 8 audit, rest fit",
        "scene_split": split,
        "selection_rule": "First state at 3.5-4.5m followed within two turns by a state at <=3m; one pair per trajectory",
        "wrong_instruction_rule": "Natural same-scene alternative goal >7m Euclidean from true goal, guaranteeing >4m from any positive state with geodesic distance <=3m",
        "counts": summaries, "per_seed": {str(seed): dict(value)
                                       for seed, value in per_seed.items()},
        "coverage_gates": gates,
        "enough_coverage_for_rgb_replay": all(gates.values()),
        "selected": {part: records[part] for part in split},
        "interpretation": "Train-only source coverage; geodesic distance labels are not model inputs, and no RGB or navigation accuracy has been tested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": summaries, "coverage_gates": gates,
                      "selected": {part: len(records[part]) for part in split}},
                     indent=2))


if __name__ == "__main__":
    main()
