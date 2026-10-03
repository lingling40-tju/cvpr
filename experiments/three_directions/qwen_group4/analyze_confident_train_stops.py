"""Pair frozen train rows and count failed voluntary STOP outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


STOP_FAILURE = "stopped but goal not reached."


def load(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 64:
        raise ValueError("expected 64 complete train steps")
    return rows, hashlib.sha256(raw).hexdigest()


def grouped(row: dict, step: int) -> dict[str, list[dict]]:
    if row["step"] != step or len(row["info"]) != 16:
        raise ValueError("step or rollout count mismatch")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in row["info"]:
        groups[str(item["episode_id"])].append(item)
    if len(groups) != 4 or set(map(len, groups.values())) != {4}:
        raise ValueError("expected four episode groups of four rollouts")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_sha = load(args.candidate)
    control, control_sha = load(args.control)
    seen = set()
    counts = {"candidate": Counter(), "control": Counter()}
    paired_deltas = Counter()
    for step, (left, right) in enumerate(zip(candidate, control), 1):
        a, b = grouped(left, step), grouped(right, step)
        if set(a) != set(b) or seen.intersection(a):
            raise ValueError("candidate/control train rows differ")
        seen.update(a)
        for episode in a:
            failed_stops = {}
            for name, items in (("candidate", a[episode]), ("control", b[episode])):
                if len({tuple(turn["response"] for turn in item["gen_traj"])
                        for item in items}) < 2:
                    raise ValueError("four rollouts were not diverse")
                for item in items:
                    if item["task_success"] != (
                            item["end_reason"] == "successfully reached the goal."):
                        raise ValueError("success/end reason mismatch")
                    counts[name][item["end_reason"]] += 1
                failed_stops[name] = sum(item["end_reason"] == STOP_FAILURE
                                         for item in items)
            paired_deltas[failed_stops["candidate"] - failed_stops["control"]] += 1
    if len(seen) != 256 or any(sum(values.values()) != 1024 for values in counts.values()):
        raise ValueError("incomplete paired group-four training coverage")
    stopped_candidate = counts["candidate"][STOP_FAILURE]
    stopped_control = counts["control"][STOP_FAILURE]
    report = {
        "schema": "confidence_train_stop_modes_v1",
        "candidate_rollout_sha256": candidate_sha,
        "control_rollout_sha256": control_sha,
        "steps": 64, "seed": 11, "group_size": 4,
        "episode_groups": len(seen), "rollouts_per_arm": 1024,
        "end_reason_counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "failed_voluntary_stop": {
            "candidate": stopped_candidate,
            "control": stopped_control,
            "difference_count": stopped_candidate - stopped_control,
            "difference_pp_of_rollouts": 100 * (stopped_candidate - stopped_control) / 1024,
            "paired_group_count_by_candidate_minus_control_stops":
                {str(k): v for k, v in sorted(paired_deltas.items())},
        },
        "interpretation": (
            "Exploratory same-row R2R-train policy-rollout diagnostic. "
            "Outcome composition and random trajectories differ across arms; "
            "this is not a causal STOP effect or a val-unseen result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
