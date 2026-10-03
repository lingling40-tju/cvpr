"""Frozen Qwen route ordinal within matched, all-failure VLN groups.

The frozen scorer ranks failures only against other failures with the same
terminal mode. All score means within each mode are zero, so no mode gets a
systematic advantage. Successful groups keep the outcome-only reward.
"""

from collections import defaultdict
import math


ELIGIBLE_MODES = (
    "stopped but goal not reached.",
    "number of turns exceeded.",
)


def group_relative_adjustments(infos, group_size):
    """Return (bonuses_to_remove, ordinal_rewards, summary) for one rollout batch."""
    if group_size != 4:
        raise ValueError("this pilot requires group size four")
    groups = defaultdict(list)
    for index, info in enumerate(infos):
        groups[str(info["episode_id"])].append(index)
    if not groups or any(len(indices) != group_size for indices in groups.values()):
        raise ValueError("rollout batch is not partitioned into matched groups of four")
    removed = []
    ordinal = [0.0] * len(infos)
    for info in infos:
        bonus = float(info["reward_components"].get("fused_bonus", 0.0))
        if not math.isfinite(bonus) or not 0 <= bonus <= 1:
            raise ValueError("invalid raw-score service bonus")
        if info["task_success"] and bonus:
            raise ValueError("success rollout unexpectedly received a semantic bonus")
        if bonus:
            scored = info.get("fused_reward", {})
            if scored.get("status") != "ok" or \
                    abs(bonus - float(scored["bonus"])) > 1e-5:
                raise ValueError("frozen score/bonus mismatch")
        removed.append(bonus)
    active_groups = 0
    eligible_pairs = 0
    for indices in groups.values():
        if any(infos[index]["task_success"] for index in indices):
            continue
        group_active = False
        for mode in ELIGIBLE_MODES:
            members = [index for index in indices
                       if infos[index]["end_reason"] == mode]
            if len(members) < 2:
                continue
            scores = []
            for index in members:
                scored = infos[index].get("fused_reward", {})
                if scored.get("status") != "ok":
                    raise ValueError("eligible failure lacks a Qwen route score")
                raw = float(scored["raw"])
                if not math.isfinite(raw):
                    raise ValueError("nonfinite frozen score")
                scores.append(raw)
            ordered = sorted(scores)
            center = (len(members) - 1) / 2
            for index, score in zip(members, scores):
                positions = [rank for rank, item in enumerate(ordered)
                             if item == score]
                average_rank = sum(positions) / len(positions)
                ordinal[index] = (average_rank - center) / (len(members) - 1)
            if abs(sum(ordinal[index] for index in members)) > 1e-6:
                raise AssertionError("mode-stratified ordinal reward is not zero-sum")
            if any(ordinal[index] for index in members):
                group_active = True
            eligible_pairs += len(members) * (len(members) - 1) // 2
        active_groups += group_active
    return removed, ordinal, {
        "matched_groups": len(groups),
        "all_failure_groups": sum(not any(infos[index]["task_success"]
                                          for index in indices)
                                  for indices in groups.values()),
        "ordinal_active_groups": active_groups,
        "same_mode_pairs": eligible_pairs,
        "bonuses_removed": sum(bonus > 0 for bonus in removed),
        "ordinal_nonzero_rollouts": sum(value != 0 for value in ordinal),
    }
