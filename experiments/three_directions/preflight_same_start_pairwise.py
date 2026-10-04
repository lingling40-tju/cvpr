"""Count train-scene same-start visual-route pairs for a relative critic.

Only existing, audited R2R-train replay labels are read. The resulting
pairs are correlated within episodes; they are not navigation outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_records(paths: list[Path]) -> tuple[dict, set[str], int]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    scenes: set[str] = set()
    record_ids: set[str] = set()
    count = 0
    for root in paths:
        for path in sorted(root.glob("*.json")):
            row = json.loads(path.read_text())
            if row["schema"] != "policy_process_turn_record_v1":
                raise ValueError(f"unexpected record schema: {path}")
            rid = row["record_id"]
            if rid in record_ids or path.stem != rid:
                raise ValueError(f"duplicate or mismatched record ID: {path}")
            record_ids.add(rid)
            distances = [row["start_distance_to_goal_for_label_only"]] + [
                turn["distance_to_goal_for_label_only"] for turn in row["turns"]]
            if not all(math.isfinite(float(d)) and d >= 0 for d in distances):
                raise ValueError(f"invalid distance trace: {path}")
            scene = row["scene_id"]
            scenes.add(scene)
            key = (scene, str(row["episode_id"]), row["instruction"])
            groups[key].append({"distances": distances, "turns": row["turns"]})
            count += 1
    if not count:
        raise ValueError("empty source records")
    return groups, scenes, count


def action_prefix(row: dict, anchor: int) -> tuple[str, ...]:
    return tuple(action for turn in row["turns"][:anchor]
                 for action in turn["motion_actions"])


def forward_meters(row: dict, anchor: int) -> float:
    cm = 0
    for action in action_prefix(row, anchor):
        if action.startswith("move forward"):
            match = re.fullmatch(r"move forward (\d+)cm", action)
            if match is None:
                raise ValueError(f"unrecognized forward action: {action}")
            cm += int(match.group(1))
    return cm / 100.0


def summarize(groups: dict, scenes: set[str], records: int) -> dict:
    report = {"records": records, "episode_groups": len(groups),
              "scenes": len(scenes), "anchors": {}}
    for anchor in (3, 6):
        pairs_all = 0
        pairs_gap1m = 0
        pairs_gap1m_distinct_actions = 0
        gaps = []
        ids: set[str] = set()
        hit_scenes: set[str] = set()
        baseline_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        baseline_ties = 0
        for (scene, eid, _), rows in groups.items():
            available = [row for row in rows if len(row["distances"]) > anchor]
            for left, right in itertools.combinations(available, 2):
                pairs_all += 1
                gap = abs(float(left["distances"][anchor])
                          - float(right["distances"][anchor]))
                if gap < 1.0:
                    continue
                pairs_gap1m += 1
                gaps.append(gap)
                ids.add(eid)
                hit_scenes.add(scene)
                pairs_gap1m_distinct_actions += (
                    action_prefix(left, anchor) != action_prefix(right, anchor))
                commanded = forward_meters(left, anchor) - forward_meters(right, anchor)
                if commanded == 0:
                    baseline_scores[(scene, eid)].append(0.5)
                    baseline_ties += 1
                else:
                    baseline_scores[(scene, eid)].append(float(
                        (commanded > 0) ==
                        (left["distances"][anchor] < right["distances"][anchor])))
        episode_means = {key: statistics.mean(scores)
                         for key, scores in baseline_scores.items()}
        scene_means: dict[str, list[float]] = defaultdict(list)
        for (scene, _), value in episode_means.items():
            scene_means[scene].append(value)
        report["anchors"][str(anchor)] = {
            "pairs_all": pairs_all,
            "pairs_gap_at_least_1m": pairs_gap1m,
            "pairs_gap_at_least_1m_with_distinct_actions":
                pairs_gap1m_distinct_actions,
            "unique_episode_ids_with_gap": len(ids),
            "scenes_with_gap": len(hit_scenes),
            "median_gap_m": statistics.median(gaps) if gaps else None,
            "forward_distance_only_baseline": {
                "ties_counted_half": baseline_ties,
                "pair_accuracy_with_half_ties": statistics.mean(
                    score for scores in baseline_scores.values() for score in scores)
                if baseline_scores else None,
                "episode_macro_accuracy": statistics.mean(episode_means.values())
                if episode_means else None,
                "scene_macro_accuracy": statistics.mean(
                    statistics.mean(values) for values in scene_means.values())
                if scene_means else None,
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-old", type=Path, required=True)
    parser.add_argument("--fit-new", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit, fit_scenes, fit_count = load_records([args.fit_old, args.fit_new])
    dev, dev_scenes, dev_count = load_records([args.development])
    if fit_scenes & dev_scenes:
        raise ValueError("fit and development scenes overlap")
    report = {
        "schema": "same_start_pairwise_progress_preflight_v1",
        "interpretation": "Train-scene label coverage only; pair counts are correlated by episode and do not establish reward or navigation gain",
        "source_sha256": {
            "old_manifest": digest(args.old_manifest),
            "new_manifest": digest(args.new_manifest),
            "policy_manifest": digest(args.policy_manifest),
        },
        "fit": summarize(fit, fit_scenes, fit_count),
        "development": summarize(dev, dev_scenes, dev_count),
        "scene_disjoint": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({part: report[part]["anchors"]
                      for part in ("fit", "development")}, sort_keys=True))


if __name__ == "__main__":
    main()
