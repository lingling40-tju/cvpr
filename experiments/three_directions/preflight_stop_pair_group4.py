"""Audit STOP-vs-continuation pair support in frozen R2R-train rollouts.

This does not estimate navigation gain or a causal benefit of continuing.
The compared rollouts share an episode but can differ before the STOP turn.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from stop_pair_group4_reward import (
    MIN_CONTINUATION_DISTANCE_M, MIN_PAIR_GAP_M, MIN_STOP_DISTANCE_M,
    STOP_FAILURE, TURN_CAP, group_relative_adjustments,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    seen = set()
    gaps = []
    step = 0
    with args.rollout.open() as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            infos = row["info"]
            if row["step"] != step or len(infos) != 16:
                raise ValueError("step or rollout count mismatch")
            groups = defaultdict(list)
            for info in infos:
                eid = str(info["episode_id"])
                groups[eid].append(info)
            if len(groups) != 4 or any(len(items) != 4 for items in groups.values()) or \
                    seen.intersection(groups):
                raise ValueError("four-rollout group or episode reuse mismatch")
            seen.update(groups)
            _, votes, summary = group_relative_adjustments(infos, 4)
            counts.update(summary)
            for info, vote in zip(infos, votes):
                if vote < 0 and info["end_reason"] != STOP_FAILURE or \
                        vote > 0 and info["end_reason"] != TURN_CAP:
                    raise ValueError("reward sign disagrees with endpoint mode")
            for items in groups.values():
                if any(item["task_success"] for item in items):
                    continue
                stop = [float(item["distance_to_goal"]) for item in items
                        if item["end_reason"] == STOP_FAILURE and
                        float(item["distance_to_goal"]) >= MIN_STOP_DISTANCE_M]
                cap = [float(item["distance_to_goal"]) for item in items
                       if item["end_reason"] == TURN_CAP and
                       float(item["distance_to_goal"]) >= MIN_CONTINUATION_DISTANCE_M]
                gaps.extend(a - b for a in stop for b in cap
                            if a - b >= MIN_PAIR_GAP_M)
    if step != 64 or len(seen) != 256 or len(gaps) != counts["continuation_over_stop_pairs"] or \
            counts["active_groups"] < 40 or len(gaps) < 100 or \
            not all(math.isfinite(gap) and gap >= MIN_PAIR_GAP_M for gap in gaps):
        raise ValueError("frozen rollout coverage or pair support gate failed")
    gaps.sort()
    result = {
        "schema": "stop_pair_group4_train_preflight_v1",
        "rollout_sha256": digest(args.rollout),
        "steps": step,
        "unique_episode_groups": len(seen),
        "rollouts": 16 * step,
        "group_size": 4,
        "rule": {
            "min_stop_distance_m": MIN_STOP_DISTANCE_M,
            "min_continuation_distance_m": MIN_CONTINUATION_DISTANCE_M,
            "min_pair_gap_m": MIN_PAIR_GAP_M,
            "pair_vote": 1 / 6,
            "all_failure_only": True,
        },
        "counts": dict(counts),
        "pair_gap_median": gaps[len(gaps) // 2],
        "interpretation": "Train-scene correlated rollout support only; no policy update or navigation gain.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
