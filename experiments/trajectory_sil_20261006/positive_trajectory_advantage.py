"""Frozen CPU-checkable advantage for a possible trajectory-selection pilot.

This is a positive-only, on-policy policy-gradient update inspired by
self-imitation. It is not the replay-buffer SIL algorithm. A rollout is
credited only when its terminal navigation quality exceeds its three peers;
poor trajectories receive zero actor advantage rather than negative credit.
The terminal score comes from the simulator's success plus nDTW reward.
"""

from __future__ import annotations

from collections import defaultdict

import torch


GROUP_SIZE = 4
SUCCESS_PRIORITY = 15.0
FAILURE_SCORE_FLOOR = 2.5
GAP_SCALE = 5.0
MAX_ADVANTAGE = 1.5


def positive_trajectory_advantage(
    terminal_score: torch.Tensor,
    task_success: torch.Tensor,
    action_mask: torch.Tensor,
    group_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign bounded positive quality gaps to generated action tokens.

    The extra success priority ensures that a successful rollout outranks
    every unsuccessful one even if distance-weighted success payoffs differ.
    A failed rollout must also earn at least half the maximum nDTW component
    before it can be imitated. Equal-quality groups receive no update.
    """
    if terminal_score.ndim != 1 or task_success.shape != terminal_score.shape:
        raise ValueError("terminal score and success must be one value per rollout")
    if action_mask.ndim != 2 or action_mask.shape[0] != len(terminal_score):
        raise ValueError("action mask dimensions differ from rollouts")
    if len(group_ids) != len(terminal_score):
        raise ValueError("group IDs do not cover the batch")
    if not torch.isfinite(terminal_score).all() or (terminal_score < 0).any():
        raise ValueError("terminal navigation scores must be finite and nonnegative")
    if not torch.all((action_mask == 0) | (action_mask == 1)):
        raise ValueError("action mask must be binary")
    if not torch.all(action_mask.sum(dim=1) > 0):
        raise ValueError("every rollout must contain generated action tokens")
    if not torch.all((task_success == 0) | (task_success == 1)):
        raise ValueError("success flags must be binary")

    groups: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        groups[str(group_id)].append(row)
    if not groups or any(len(rows) != GROUP_SIZE for rows in groups.values()):
        raise ValueError("every navigation instruction needs four rollouts")

    score = terminal_score.float()
    success = task_success.bool()
    quality = score + SUCCESS_PRIORITY * success.float()
    advantage = torch.zeros_like(action_mask, dtype=score.dtype)
    for rows in groups.values():
        for row in rows:
            peer_mean = (quality[rows].sum() - quality[row]) / (GROUP_SIZE - 1)
            gap = (quality[row] - peer_mean).clamp(min=0.0)
            eligible = bool(success[row]) or float(score[row]) >= FAILURE_SCORE_FLOOR
            if eligible:
                weight = (gap / GAP_SCALE).clamp(max=MAX_ADVANTAGE)
                advantage[row, action_mask[row].bool()] = weight
    return advantage, advantage.clone()


def synthetic_self_check() -> None:
    mask = torch.tensor([[1, 1, 0]] * 8)
    scores = torch.tensor([8.0, 4.0, 3.0, 0.0, 3.0, 2.0, 1.0, 0.0])
    success = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    adv, ret = positive_trajectory_advantage(
        scores, success, mask, ["first"] * 4 + ["second"] * 4,
    )
    assert torch.equal(adv, ret)
    assert adv[0, 0] == MAX_ADVANTAGE and adv[0, 1] == adv[0, 0]
    assert adv[4, 0] > 0  # high-quality failure can still teach navigation
    assert torch.count_nonzero(adv[1:4]) == 0
    assert torch.count_nonzero(adv[5:]) == 0
    assert torch.count_nonzero(adv[:, 2]) == 0  # observation/padding tokens
    tied, _ = positive_trajectory_advantage(
        torch.full((4,), 3.0), torch.zeros(4), mask[:4], ["tie"] * 4,
    )
    assert torch.count_nonzero(tied) == 0


if __name__ == "__main__":
    synthetic_self_check()
    print("positive trajectory advantage invariants passed")
