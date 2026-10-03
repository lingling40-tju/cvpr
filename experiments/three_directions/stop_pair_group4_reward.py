"""Pairwise premature-STOP credit for four-rollout all-failure groups.

This is a train-only simulator-distance mechanism diagnostic. It uses no
Qwen score and cannot be described as a learned semantic reward. A capped
continuation is preferred only when it remains outside the success radius
and finishes at least one meter nearer than a failed STOP from the same
episode. Successful groups keep the ordinary outcome reward.
"""

from __future__ import annotations

from collections import defaultdict
import math


STOP_FAILURE = "stopped but goal not reached."
TURN_CAP = "number of turns exceeded."
MIN_STOP_DISTANCE_M = 3.5
MIN_CONTINUATION_DISTANCE_M = 3.0
MIN_PAIR_GAP_M = 1.0


def group_relative_adjustments(infos: list[dict], group_size: int):
    """Return (removed_bonus, bounded_pair_vote, audit_counts)."""
    if group_size != 4:
        raise ValueError("premature-STOP pilot requires group size four")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(infos):
        groups[str(info["episode_id"])].append(index)
    if not groups or any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("rollouts are not matched four-sample episode groups")

    distances = []
    for info in infos:
        distance = float(info["distance_to_goal"])
        bonus = float(info["reward_components"].get("fused_bonus", 0.0))
        if not math.isfinite(distance) or distance < 0 or bonus != 0:
            raise ValueError("invalid distance or unexpected teacher bonus")
        distances.append(distance)

    ordinal = [0.0] * len(infos)
    counts = {
        "matched_groups": len(groups),
        "all_failure_groups": 0,
        "mixed_eligible_groups": 0,
        "active_groups": 0,
        "continuation_over_stop_pairs": 0,
        "nonzero_rollouts": 0,
    }
    for indices in groups.values():
        if any(bool(infos[index]["task_success"]) for index in indices):
            continue
        counts["all_failure_groups"] += 1
        stops = [index for index in indices
                 if infos[index]["end_reason"] == STOP_FAILURE and
                 distances[index] >= MIN_STOP_DISTANCE_M]
        continuations = [index for index in indices
                         if infos[index]["end_reason"] == TURN_CAP and
                         distances[index] >= MIN_CONTINUATION_DISTANCE_M]
        if stops and continuations:
            counts["mixed_eligible_groups"] += 1
        pairs = [(stop, continuation)
                 for stop in stops for continuation in continuations
                 if distances[stop] - distances[continuation] >= MIN_PAIR_GAP_M]
        for stop, continuation in pairs:
            ordinal[stop] -= 1.0 / 6.0
            ordinal[continuation] += 1.0 / 6.0
        counts["continuation_over_stop_pairs"] += len(pairs)
        counts["active_groups"] += bool(pairs)
        if abs(sum(ordinal[index] for index in indices)) > 1e-7:
            raise AssertionError("pairwise reward is not group zero-sum")
    if any(abs(value) > 0.5 + 1e-9 or not math.isfinite(value)
           for value in ordinal):
        raise ValueError("pairwise reward exceeds its bound")
    counts["nonzero_rollouts"] = sum(value != 0 for value in ordinal)
    return [0.0] * len(infos), ordinal, counts
