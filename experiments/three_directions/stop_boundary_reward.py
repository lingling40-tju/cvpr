"""Training-only privileged stop-boundary turn reward.

Simulator distance supplies a mechanism upper bound. It is never a policy
observation or an observation-only learned semantic reward.
"""

from __future__ import annotations

import math


GOAL_RADIUS_M = 3.0
CONTINUE_PENALTY = 0.1


def boundary_turn_reward(start_m: float, before_m: float, after_m: float,
                         executed_actions: list[str], stop_in_response: bool) -> float:
    """Potential change outside the success radius, then penalize continuing.

    A response containing STOP gets no auxiliary credit even if the simulator
    executed earlier commands from that same response, matching the oracle
    baseline's STOP handling. All non-STOP executed actions, including turns,
    count as continuing navigation inside the goal region.
    """
    distances = (start_m, before_m, after_m)
    if any(not math.isfinite(value) or value < 0 for value in distances):
        raise ValueError("invalid simulator distance")
    if not isinstance(executed_actions, list) or \
            any(not isinstance(action, str) for action in executed_actions):
        raise ValueError("invalid executed actions")
    if stop_in_response or not executed_actions:
        return 0.0
    scale = max(start_m, GOAL_RADIUS_M)
    change = (max(before_m - GOAL_RADIUS_M, 0.0) -
              max(after_m - GOAL_RADIUS_M, 0.0)) / scale
    penalty = CONTINUE_PENALTY if before_m <= GOAL_RADIUS_M else 0.0
    return change - penalty


def synthetic_self_check() -> dict:
    move = ["move forward 25cm"]
    assert boundary_turn_reward(10, 5, 4, move, False) == .1
    assert boundary_turn_reward(10, 4, 2, move, False) == .1
    assert boundary_turn_reward(10, 2, 1, move, False) == -.1
    assert boundary_turn_reward(10, 2, 4, move, False) == -.2
    assert boundary_turn_reward(10, 2, 1, move, True) == 0.0
    assert boundary_turn_reward(10, 2, 1, [], False) == 0.0
    for invalid in (float("nan"), float("inf"), -1.0):
        try:
            boundary_turn_reward(10, invalid, 2, move, False)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite or negative distance accepted")
    return {"outside_progress": True, "inside_continuation_penalized": True,
            "stop_auxiliary_zero": True, "invalid_distance_rejected": True}


if __name__ == "__main__":
    print(synthetic_self_check())
