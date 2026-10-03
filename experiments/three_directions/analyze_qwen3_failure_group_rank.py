"""Check whether frozen Qwen route scores order failed trajectories.

Simulator distance is used only as a label for this offline screen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-analysis", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher_analysis.read_text())
    selected = json.loads(args.policy_manifest.read_text())
    group = json.loads(args.group_manifest.read_text())
    if teacher["schema"] != "qwen3_terminal_route_analysis_v1" or \
            teacher["part"] != "fit" or not teacher["gate"]["passed"] or \
            selected["schema"] != "qwen3_policy_route_manifest_v1" or \
            teacher["source_sha256"]["policy_manifest"] != digest(args.policy_manifest) or \
            selected["source_sha256"]["group_manifest"] != digest(args.group_manifest) or \
            group["schema"] != "policy_group_relative_manifest_v1":
        raise ValueError("frozen teacher/source mismatch")
    source = {(r["seed"], str(r["episode_id"]), r["variant"]): r
              for r in group["selected"]["fit"]}
    plan = {(g["seed"], g["episode_id"]): g
            for g in selected["selected"]["fit"]}
    if len(plan) != 55 or len(teacher["per_group"]) != 55:
        raise ValueError("fit group coverage changed")
    rows = {"same_mode": [], "cross_mode": []}
    by_group = defaultdict(list)
    by_scene = defaultdict(list)
    mode_pairs = Counter()
    all_failure_groups = 0
    for scored in teacher["per_group"]:
        key = (scored["seed"], scored["episode_id"])
        manifest_group = plan[key]
        if scored["scene_id"] != manifest_group["scene_id"] or \
                len(scored["routes"]) != 4:
            raise ValueError("group identity mismatch")
        if any(r["success"] for r in scored["routes"]):
            continue
        all_failure_groups += 1
        entries = []
        for scored_route, manifest_route in zip(scored["routes"], manifest_group["routes"]):
            if scored_route["record_id"] != manifest_route["record_id"]:
                raise ValueError("route identity mismatch")
            label = source[(scored["seed"], scored["episode_id"],
                            manifest_route["variant"])]
            if label["terminal_mode"] != scored_route["terminal_mode"]:
                raise ValueError("outcome label mismatch")
            entries.append({"record_id": scored_route["record_id"],
                            "mode": scored_route["terminal_mode"],
                            "distance": float(label["terminal_distance_m_for_replay_audit_only"]),
                            "teacher_score": float(scored_route["terminal_margin"])})
        for a, b in itertools.combinations(entries, 2):
            gap = abs(a["distance"] - b["distance"])
            if gap < 1.5:
                continue
            better, worse = (a, b) if a["distance"] < b["distance"] else (b, a)
            same = better["mode"] == worse["mode"]
            bucket = "same_mode" if same else "cross_mode"
            correct = int(better["teacher_score"] > worse["teacher_score"])
            row = {"seed": scored["seed"], "episode_id": scored["episode_id"],
                   "scene_id": scored["scene_id"],
                   "better_record": better["record_id"],
                   "worse_record": worse["record_id"],
                   "distance_gap_m_for_audit_only": gap,
                   "teacher_score_difference": better["teacher_score"] - worse["teacher_score"],
                   "better_terminal_mode": better["mode"],
                   "worse_terminal_mode": worse["mode"],
                   "correct": correct}
            rows[bucket].append(row)
            mode_pairs[(better["mode"], worse["mode"])] += 1
            if same:
                group_key = f"{scored['seed']}:{scored['episode_id']}"
                by_group[group_key].append(correct)
                by_scene[scored["scene_id"]].append(correct)
    if all_failure_groups != 33 or len(rows["same_mode"]) != 80 or \
            len(rows["cross_mode"]) != 67 or len(by_group) != 30:
        raise ValueError("frozen failure-pair denominator changed")
    rng = random.Random(11)
    groups = list(by_group.values())
    boots = []
    for _ in range(2000):
        sample = [rng.choice(groups) for _ in groups]
        boots.append(sum(map(sum, sample)) / sum(map(len, sample)))
    boots.sort()
    correct = sum(r["correct"] for r in rows["same_mode"])
    group_macro = sum(sum(v) / len(v) for v in by_group.values()) / len(by_group)
    metrics = {"same_mode_correct": correct, "same_mode_pairs": 80,
               "same_mode_accuracy": correct / 80,
               "same_mode_group_macro": group_macro,
               "same_mode_scene_macro": sum(sum(v) / len(v) for v in by_scene.values()) /
                                        len(by_scene),
               "same_mode_group_bootstrap95": [boots[50], boots[1949]],
               "cross_mode_correct": sum(r["correct"] for r in rows["cross_mode"]),
               "cross_mode_pairs": 67,
               "mode_pair_counts": {f"{a} | {b}": n for (a, b), n in mode_pairs.items()}}
    passed = correct >= 56 and group_macro >= 0.65
    report = {"schema": "qwen3_failure_group_rank_v1",
              "interpretation": "exploratory frozen train-rollout rank; simulator distance is label-only",
              "source_sha256": {"teacher_analysis": digest(args.teacher_analysis),
                                "policy_manifest": digest(args.policy_manifest),
                                "group_manifest": digest(args.group_manifest)},
              "coverage": {"all_failure_groups": all_failure_groups,
                           "same_mode_groups": len(by_group),
                           "same_mode_scenes": len(by_scene)},
              "metrics": metrics,
              "gate": {"same_mode_correct_min": 56,
                       "same_mode_group_macro_min": 0.65,
                       "passed": passed},
              "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage": report["coverage"], "metrics": metrics,
                      "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
