"""Pure helpers for the frozen outcome-consistent RLOO pilot."""
from typing import Tuple

REASON_SUCCESS = "successfully reached the goal."
REASON_GOAL_NOT_REACHED = "stopped but goal not reached."


def forced_stop_outcome(oracle_success: bool) -> Tuple[bool, str]:
    """Resolve a hard budget truncation exactly as the evaluator's forced STOP."""
    success = bool(oracle_success)
    return success, REASON_SUCCESS if success else REASON_GOAL_NOT_REACHED


def terminal_path_credit(reason: str, success: bool, ndtw: float, weight: float,
                         success_only: bool) -> float:
    """nDTW credit is zero at every failed terminal when success_only is frozen."""
    if reason not in (REASON_SUCCESS, REASON_GOAL_NOT_REACHED):
        return 0.0
    if success_only and not success:
        return 0.0
    return float(ndtw) * float(weight)
