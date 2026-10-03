"""Candidate n=4 reward from high-margin Qwen pair preferences.

This is a staged algorithm, not the reward in the ongoing ungated pilot.
The 5.5 logit-margin gap is the upper median of absolute same-mode
failure-pair gaps on the frozen fit cache. The diagnostic was post hoc,
so a separate matched navigation experiment is required before use.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import math


ELIGIBLE_MODES = (
    "stopped but goal not reached.",
    "number of turns exceeded.",
)
MIN_SCORE_GAP = 5.5


def group_relative_adjustments(infos, group_size):
    """Return (removed_bonus, bounded_zero_sum_pair_vote, counts)."""
    if group_size != 4:
        raise ValueError("confidence-pair diagnostic uses standard group size four")
    groups = defaultdict(list)
    for index, info in enumerate(infos):
        groups[str(info["episode_id"])].append(index)
    if not groups or any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("rollouts are not four-sample matched episode groups")

    removed = []
    for info in infos:
        bonus = float(info["reward_components"].get("fused_bonus", 0.0))
        if not math.isfinite(bonus) or not 0 <= bonus <= 1:
            raise ValueError("invalid teacher-service bonus")
        if info["task_success"] and bonus:
            raise ValueError("successful rollout received teacher bonus")
        if bonus:
            scored = info.get("fused_reward", {})
            if scored.get("status") != "ok" or \
                    abs(bonus - float(scored["bonus"])) > 1e-5:
                raise ValueError("teacher bonus/score mismatch")
        removed.append(bonus)

    ordinal = [0.0] * len(infos)
    all_failure_groups = 0
    active_groups = 0
    same_mode_pairs = 0
    confident_pairs = 0
    for indices in groups.values():
        if any(infos[index]["task_success"] for index in indices):
            continue
        all_failure_groups += 1
        for mode in ELIGIBLE_MODES:
            members = [index for index in indices
                       if infos[index]["end_reason"] == mode]
            if len(members) < 2:
                continue
            scores = {}
            for index in members:
                scored = infos[index].get("fused_reward", {})
                if scored.get("status") != "ok":
                    raise ValueError("eligible failure lacks a Qwen score")
                raw = float(scored["raw"])
                if not math.isfinite(raw):
                    raise ValueError("nonfinite Qwen score")
                scores[index] = raw
            same_mode_pairs += len(members) * (len(members) - 1) // 2
            scale = 2 * (len(members) - 1)
            for left, right in itertools.combinations(members, 2):
                gap = scores[left] - scores[right]
                if abs(gap) < MIN_SCORE_GAP:
                    continue
                sign = 1.0 if gap > 0 else -1.0
                ordinal[left] += sign / scale
                ordinal[right] -= sign / scale
                confident_pairs += 1
            if abs(sum(ordinal[index] for index in members)) > 1e-6:
                raise AssertionError("confidence-pair reward is not zero-sum")
        active_groups += any(ordinal[index] != 0 for index in indices)
    if any(not math.isfinite(value) or abs(value) > 0.5 + 1e-9
           for value in ordinal):
        raise ValueError("confidence-pair reward exceeds its bound")
    return removed, ordinal, {
        "matched_groups": len(groups),
        "all_failure_groups": all_failure_groups,
        "ordinal_active_groups": active_groups,
        "same_mode_pairs": same_mode_pairs,
        "confident_pairs": confident_pairs,
        "bonuses_removed": sum(value > 0 for value in removed),
        "ordinal_nonzero_rollouts": sum(value != 0 for value in ordinal),
        "min_score_gap": MIN_SCORE_GAP,
    }
