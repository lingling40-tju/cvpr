"""Exploratory paired failure-mode summary for the 64-step oracle recheck.

This analysis is post hoc. It describes checkpoint behavior and cannot
identify whether turn-wise progress credit caused a particular action.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent / "ordinal_progress/policy_preference"
EPISODES = ROOT / "paired_oracle_full_episodes.jsonl"
ANALYSIS = ROOT / "oracle_full_recheck_analysis.json"
SCREEN = ROOT / "process_val256_manifest.json"
OUTPUT = ROOT / "oracle_full_failure_modes.json"
SCREEN_SHA256 = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"


def summary(rows: list[dict]) -> dict:
    patterns = Counter()
    max_turns = Counter()
    max_turns_pairs = Counter()
    distances = []
    both_failed_distances = []
    scenes = defaultdict(lambda: {"episodes": 0, "candidate_successes": 0,
                                  "control_successes": 0})
    for row in rows:
        candidate, control = row["candidate"], row["control"]
        pattern = ("success" if candidate["success"] else "failure") + "_vs_" + (
            "success" if control["success"] else "failure")
        patterns[pattern] += 1
        c_turn_cap = candidate["early_stop_reason"] == "max_turns_reached"
        b_turn_cap = control["early_stop_reason"] == "max_turns_reached"
        max_turns["candidate"] += c_turn_cap
        max_turns["control"] += b_turn_cap
        max_turns_pairs[("candidate" if c_turn_cap else "no_candidate") + "_" +
                        ("control" if b_turn_cap else "no_control")] += 1
        delta = candidate["distance_to_goal_m"] - control["distance_to_goal_m"]
        distances.append(delta)
        if not candidate["success"] and not control["success"]:
            both_failed_distances.append(delta)
        scene = scenes[row["scene_id"]]
        scene["episodes"] += 1
        scene["candidate_successes"] += bool(candidate["success"])
        scene["control_successes"] += bool(control["success"])
    return {
        "episodes": len(rows),
        "success_patterns": dict(sorted(patterns.items())),
        "max_turns_reached": dict(max_turns),
        "paired_max_turns_reached": dict(sorted(max_turns_pairs.items())),
        "mean_candidate_minus_control_terminal_distance_m": statistics.mean(distances),
        "both_failed_mean_terminal_distance_delta_m": statistics.mean(both_failed_distances),
        "scenes_candidate_better_equal_worse": {
            "better": sum(s["candidate_successes"] > s["control_successes"]
                          for s in scenes.values()),
            "equal": sum(s["candidate_successes"] == s["control_successes"]
                         for s in scenes.values()),
            "worse": sum(s["candidate_successes"] < s["control_successes"]
                         for s in scenes.values()),
        },
    }


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    if hashlib.sha256(SCREEN.read_bytes()).hexdigest() != SCREEN_SHA256:
        raise ValueError("screen manifest hash changed")
    screen = {str(x) for x in json.loads(SCREEN.read_text())["episode_ids"]}
    rows = [json.loads(line) for line in EPISODES.read_text().splitlines()]
    ids = [str(row["episode_id"]) for row in rows]
    if len(rows) != 1839 or len(set(ids)) != 1839 or len(screen) != 256:
        raise ValueError("paired episode or screen coverage changed")
    if (sum(bool(row["candidate"]["success"]) for row in rows)
            != analysis["full"]["candidate_metrics"]["successes"]
            or sum(bool(row["control"]["success"]) for row in rows)
            != analysis["full"]["control_metrics"]["successes"]):
        raise ValueError("episode outcomes disagree with frozen full analysis")
    outside = [row for row in rows if str(row["episode_id"]) not in screen]
    if len(outside) != 1583:
        raise ValueError("outside-screen complement changed")
    if (sum(bool(row["candidate"]["success"]) for row in outside)
            != analysis["outside_reused_screen"]["candidate_metrics"]["successes"]
            or sum(bool(row["control"]["success"]) for row in outside)
            != analysis["outside_reused_screen"]["control_metrics"]["successes"]):
        raise ValueError("screen complement outcomes disagree with analysis")
    result = {
        "schema": "oracle_64_posthoc_failure_modes_v1",
        "interpretation": "Post-screen descriptive analysis; no causal or learned-reward claim",
        "sources_sha256": {
            "episodes": hashlib.sha256(EPISODES.read_bytes()).hexdigest(),
            "full_analysis": hashlib.sha256(ANALYSIS.read_bytes()).hexdigest(),
            "screen_manifest": SCREEN_SHA256,
        },
        "full": summary(rows),
        "outside_reused_screen": summary(outside),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
