"""CPU-only coverage inventory reusing audited n=4 oracle train rollouts.

This conditional, exploratory extension uses already scheduled seeds. It
never creates cross-seed trajectory pairs or counts the same episode as
multiple independent groups. Simulator distances are labels only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path

from preflight_group_future_advantage import (
    EXPECTED_DATASET, EXPECTED_SPLIT, commanded_forward_meters, digest,
    point, summarize,
)


SEEDS = (11, 22, 33)
ANCHORS = (3, 6)
GROUPS_PER_STEP = 4
ROLLOUTS_PER_GROUP = 4
STEPS = 128
EPISODES = STEPS * GROUPS_PER_STEP


def collect_seed(rollout: Path, episodes: dict, scene_to_part: dict,
                 *, expected_steps: int = STEPS) -> tuple[dict, dict, dict, list]:
    """Return within-seed pairs, all-failure IDs, and ordered training rows."""
    rows = {part: {anchor: [] for anchor in ANCHORS}
            for part in set(scene_to_part.values())}
    same_mode = {part: {anchor: [] for anchor in ANCHORS}
                 for part in rows}
    all_failure = {part: set() for part in rows}
    seen = set()
    ordered_rows = []
    with rollout.open() as stream:
        for step, line in enumerate(stream, 1):
            batch = json.loads(line)
            if step > expected_steps or batch.get("step") != step or \
                    len(batch.get("info", [])) != 16:
                raise ValueError(f"invalid rollout step {step}: {rollout}")
            groups = defaultdict(list)
            for info in batch["info"]:
                groups[str(info["episode_id"])].append(info)
            if len(groups) != GROUPS_PER_STEP or any(
                    len(group) != ROLLOUTS_PER_GROUP
                    for group in groups.values()):
                raise ValueError(f"not group size four at step {step}")
            if seen.intersection(groups):
                raise ValueError("episode repeated within a seed")
            seen.update(groups)
            ordered_rows.append(sorted(groups))
            for eid, infos in groups.items():
                if eid not in episodes:
                    raise ValueError(f"non-train episode: {eid}")
                scene = str(episodes[eid]["scene_id"])
                part = scene_to_part.get(scene)
                if part is None:
                    raise ValueError(f"scene absent from fixed split: {scene}")
                if any(bool(info["task_success"]) for info in infos):
                    continue
                all_failure[part].add(eid)
                for anchor in ANCHORS:
                    active = [info for info in infos
                              if len(info["gen_traj"]) > anchor]
                    for left, right in itertools.combinations(active, 2):
                        lturns, rturns = left["gen_traj"], right["gen_traj"]
                        gap = (sum(float(t["oracle_turn_progress"])
                                   for t in lturns[anchor:]) -
                               sum(float(t["oracle_turn_progress"])
                                   for t in rturns[anchor:]))
                        if not math.isfinite(gap):
                            raise ValueError("nonfinite future return gap")
                        if abs(gap) < .25:
                            continue
                        lpast = sum(float(t["oracle_turn_progress"])
                                    for t in lturns[:anchor])
                        rpast = sum(float(t["oracle_turn_progress"])
                                    for t in rturns[:anchor])
                        pair = {
                            "scene": scene, "episode_id": eid,
                            "future_gap": gap,
                            "forward_prefix": point(
                                commanded_forward_meters(lturns, anchor),
                                commanded_forward_meters(rturns, anchor), gap),
                            "oracle_past_progress": point(lpast, rpast, gap),
                        }
                        rows[part][anchor].append(pair)
                        if left["end_reason"] == right["end_reason"]:
                            same_mode[part][anchor].append(pair)
    if len(ordered_rows) != expected_steps or \
            len(seen) != expected_steps * GROUPS_PER_STEP:
        raise ValueError(f"incomplete n=4 rollout: {rollout}")
    return rows, same_mode, all_failure, ordered_rows


def coverage_checks(report: dict) -> dict:
    checks = {}
    for anchor in ANCHORS:
        key = str(anchor)
        fit = report["parts"]["fit"]["anchors"][key]
        dev = report["parts"]["development"]["anchors"][key]
        fit_same = report["parts"]["fit"]["same_terminal_mode_anchors"][key]
        dev_same = report["parts"]["development"]["same_terminal_mode_anchors"][key]
        checks[f"fit_anchor{anchor}_groups_at_least_150"] = (
            fit["episode_groups"] >= 150)
        checks[f"dev_anchor{anchor}_groups_at_least_35"] = (
            dev["episode_groups"] >= 35)
        checks[f"dev_anchor{anchor}_pairs_at_least_120"] = (
            dev["pairs"] >= 120)
        checks[f"fit_anchor{anchor}_same_mode_groups_at_least_100"] = (
            fit_same["episode_groups"] >= 100)
        checks[f"dev_anchor{anchor}_same_mode_groups_at_least_25"] = (
            dev_same["episode_groups"] >= 25)
        checks[f"dev_anchor{anchor}_same_mode_pairs_at_least_60"] = (
            dev_same["pairs"] >= 60)
    return checks


def inventory(runs: list[tuple[int, Path, Path]], dataset: Path,
              scene_split: Path, *, expected_steps: int = STEPS) -> dict:
    if len({seed for seed, _, _ in runs}) != len(runs) or \
            any(seed not in SEEDS for seed, _, _ in runs) or not runs:
        raise ValueError("expected distinct oracle seeds 11, 22, 33")
    if digest(dataset) != EXPECTED_DATASET or \
            digest(scene_split) != EXPECTED_SPLIT:
        raise ValueError("frozen train dataset or scene split changed")
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(stream)["episodes"]}
    split = json.loads(scene_split.read_text())["scene_split"]
    scene_to_part = {scene: part for part, scenes in split.items()
                     for scene in scenes}
    if len(scene_to_part) != sum(map(len, split.values())) or \
            "fit" not in split or "development" not in split:
        raise ValueError("invalid scene partition")
    rows = {part: {anchor: [] for anchor in ANCHORS} for part in split}
    same_mode = {part: {anchor: [] for anchor in ANCHORS} for part in split}
    all_failure = {part: set() for part in split}
    sources = {}
    first_order = None
    for seed, rollout, audit_path in sorted(runs):
        audit = json.loads(audit_path.read_text())
        if audit.get("schema") != "oracle_turnwise_exact512_scale_train_audit_v1" or \
                audit.get("seed") != seed or audit.get("steps") != STEPS or \
                audit.get("group_size") != 4 or \
                audit.get("unique_train_episodes") != EPISODES or \
                audit.get("same_row_candidate_control") is not True or \
                audit.get("nonzero_actor_gradient_steps", {}).get(
                    "candidate") != STEPS:
            raise ValueError(f"incomplete seed {seed} training audit")
        source_rows, source_same, source_fail, order = collect_seed(
            rollout, episodes, scene_to_part, expected_steps=expected_steps)
        if first_order is None:
            first_order = order
        elif order != first_order:
            raise ValueError("training episode rows differ across seeds")
        for part in split:
            all_failure[part].update(source_fail[part])
            for anchor in ANCHORS:
                rows[part][anchor].extend(source_rows[part][anchor])
                same_mode[part][anchor].extend(source_same[part][anchor])
        sources[str(seed)] = {"rollout": digest(rollout),
                              "train_audit": digest(audit_path)}
    report = {
        "schema": "group_future_advantage_exact512_pooled_preflight_v1",
        "seeds": sorted(seed for seed, _, _ in runs),
        "unique_train_episodes": expected_steps * GROUPS_PER_STEP,
        "group_size": ROLLOUTS_PER_GROUP,
        "steps_per_seed": expected_steps,
        "source_sha256": {"dataset": digest(dataset),
                          "scene_split": digest(scene_split),
                          "seeds": sources},
        "definition": (
            "All-failure n=4 within-seed, same-episode pairs active past "
            "anchor with abs future oracle return gap >=0.25; pooled "
            "across seeds by unique episode ID, never cross-seed pairs"
        ),
        "parts": {part: {
            "all_failure_episode_groups": len(all_failure[part]),
            "anchors": {str(anchor): summarize(rows[part][anchor])
                        for anchor in ANCHORS},
            "same_terminal_mode_anchors": {
                str(anchor): summarize(same_mode[part][anchor])
                for anchor in ANCHORS},
        } for part in split},
        "interpretation": (
            "Exploratory train-scene label coverage, using only already "
            "scheduled group-four rollouts. It is not an independent "
            "semantic audit, learned reward, or navigation gain."
        ),
    }
    report["coverage_checks"] = coverage_checks(report)
    report["enough_coverage_for_fit_preparation"] = all(
        report["coverage_checks"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("SEED", "ROLLOUT", "TRAIN_AUDIT"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [(int(seed), Path(rollout), Path(audit))
            for seed, rollout, audit in args.run]
    report = inventory(runs, args.dataset, args.scene_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seeds": report["seeds"],
                      "coverage_checks": report["coverage_checks"],
                      "enough_coverage_for_fit_preparation":
                      report["enough_coverage_for_fit_preparation"]}, indent=2))


if __name__ == "__main__":
    main()
