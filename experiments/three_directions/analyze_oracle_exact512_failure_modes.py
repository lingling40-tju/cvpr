"""Describe paired endpoint and termination patterns in exact512 exports.

This is post hoc behavior analysis. It does not establish that the privileged
turn-wise reward caused any termination pattern or navigation outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    outcomes = Counter()
    terminations = {arm: Counter() for arm in ("candidate", "control")}
    near_miss = Counter()
    distance_deltas = []
    both_failed = []
    by_scene = defaultdict(lambda: Counter())
    for row in rows:
        candidate, control = row["candidate"], row["control"]
        outcomes[(bool(candidate["success"]), bool(control["success"]))] += 1
        for arm, value in (("candidate", candidate), ("control", control)):
            terminations[arm][str(value["early_stop_reason"])] += 1
            if not value["success"] and 3 <= value["distance_to_goal_m"] < 4:
                near_miss[arm] += 1
            by_scene[row["scene_id"]][arm] += bool(value["success"])
        delta = candidate["distance_to_goal_m"] - control["distance_to_goal_m"]
        distance_deltas.append(delta)
        if not candidate["success"] and not control["success"]:
            both_failed.append(delta)
    return {
        "episodes": len(rows),
        "success_patterns": {
            "both_success": outcomes[(True, True)],
            "candidate_only": outcomes[(True, False)],
            "control_only": outcomes[(False, True)],
            "both_fail": outcomes[(False, False)],
        },
        "termination_reasons": {arm: dict(sorted(value.items()))
                                for arm, value in terminations.items()},
        "failed_terminal_distance_3_to_4m": dict(near_miss),
        "mean_candidate_minus_control_terminal_distance_m":
            statistics.mean(distance_deltas),
        "both_failed_mean_terminal_distance_delta_m":
            statistics.mean(both_failed) if both_failed else None,
        "scenes_candidate_better_equal_worse": {
            name: sum((value["candidate"] > value["control"]) if name == "better"
                      else (value["candidate"] == value["control"]) if name == "equal"
                      else (value["candidate"] < value["control"])
                      for value in by_scene.values())
            for name in ("better", "equal", "worse")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.full_manifest) != FULL_SHA or \
            digest(args.screen_manifest) != SCREEN_SHA or \
            len(args.seeds) != len(set(args.seeds)) or \
            any(seed not in (11, 22, 33) for seed in args.seeds):
        raise ValueError("frozen manifest or seeds changed")
    full = json.loads(args.full_manifest.read_text())
    ids = [str(eid) for eid in full["episode_ids"]]
    scenes = [str(scene) for scene in full["scene_ids"]]
    screen_ids = {str(eid) for eid in json.loads(
        args.screen_manifest.read_text())["episode_ids"]}
    if len(ids) != len(set(ids)) or len(ids) != 1839 or \
            len(scenes) != len(ids) or len(screen_ids) != 256 or \
            not screen_ids.issubset(ids):
        raise ValueError("manifest coverage changed")
    reports = {}
    for seed in args.seeds:
        stem = f"seed{seed}_full1839"
        path = args.package / f"{stem}_paired_episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != len(ids) or \
                [row["episode_id"] for row in rows] != ids or \
                [row["scene_id"] for row in rows] != scenes:
            raise ValueError(f"seed {seed}: episode or scene mismatch")
        analysis_path = args.package / f"{stem}_early_analysis.json"
        analysis = json.loads(analysis_path.read_text())
        outside = [row for row in rows if row["episode_id"] not in screen_ids]
        if len(outside) != 1583:
            raise ValueError("screen complement changed")
        for scope, subset in (("full", rows),
                              ("outside_reused_screen", outside)):
            for arm in ("candidate", "control"):
                if sum(row[arm]["success"] for row in subset) != \
                        analysis[scope][f"{arm}_metrics"]["successes"]:
                    raise ValueError(f"seed {seed}: {scope} {arm} count mismatch")
        reports[str(seed)] = {
            "episode_export_sha256": digest(path),
            "paired_analysis_sha256": digest(analysis_path),
            "full": summarize(rows),
            "outside_reused_screen": summarize(outside),
        }
    result = {
        "schema": "oracle_exact512_posthoc_failure_modes_v1",
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "seeds": args.seeds,
        "reports": reports,
        "interpretation": "Post hoc checkpoint behavior; no causal attribution or learned-reward result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({seed: {"full": value["full"]}
                      for seed, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
