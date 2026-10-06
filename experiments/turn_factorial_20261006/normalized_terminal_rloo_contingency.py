"""CPU-checkable contingency: token-scale-matched turn-level RLOO.

This module is not used by the active four-arm run. It fixes a scale
confound found before that run's navigation outcomes: the active RLOO
adapter divides each turn's advantage by its action-token count, whereas
the GRPO control assigns an order-one advantage to each action token.
Only terminal outcome feedback is supported here; process rewards are
excluded to avoid amplifying tiny simulator-progress differences.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def normalized_terminal_turn_rloo(
    outcome_reward: torch.Tensor,
    turn_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    group_ids: list[str],
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an active-peer LOO contrast standardized per turn.

    The terminal return is the only reward. At each turn, compare only
    rollouts still active. Each generated action token receives the same
    standardized turn contrast, matching the control's token weighting.
    Observation tokens have turn index -1 and zero advantage.
    """
    if outcome_reward.ndim != 1 or turn_valid.ndim != 2 or action_turn_index.ndim != 2:
        raise ValueError("expected outcome vector, turn mask, and token-turn map")
    batch_size, turn_count = turn_valid.shape
    if len(outcome_reward) != batch_size or action_turn_index.shape[0] != batch_size or len(group_ids) != batch_size:
        raise ValueError("batch dimensions differ")
    if epsilon <= 0 or not torch.isfinite(outcome_reward).all():
        raise ValueError("invalid epsilon or terminal outcome")
    if ((action_turn_index < -1) | (action_turn_index >= turn_count)).any():
        raise ValueError("action token maps outside turn range")
    valid = turn_valid.bool()
    for row in range(batch_size):
        present = {int(x) for x in action_turn_index[row].tolist() if x >= 0}
        expected = set(torch.nonzero(valid[row], as_tuple=False).flatten().tolist())
        if not expected or expected != set(range(len(expected))) or present != expected:
            raise ValueError(f"turn-token alignment failed for rollout {row}")
    groups: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        groups[str(group_id)].append(row)
    if any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("each group must have four rollouts")

    # Gamma=1: every still-active turn sees the final outcome return.
    advantages = torch.zeros(action_turn_index.shape, dtype=outcome_reward.dtype,
                             device=outcome_reward.device)
    for rows in groups.values():
        for turn in range(turn_count):
            active = [row for row in rows if bool(valid[row, turn])]
            if len(active) < 2:
                continue
            values = outcome_reward[active] / 15.0
            n = len(active)
            loo = (n * values - values.sum()) / (n - 1)
            scale = loo.std(unbiased=True)
            if float(scale) <= epsilon:
                continue
            for row, contrast in zip(active, loo / scale):
                advantages[row, action_turn_index[row] == turn] = contrast
    return advantages, advantages.clone()


def synthetic_self_check() -> None:
    outcome = torch.tensor([15.0, 0.0, 0.0, 0.0])
    valid = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=torch.bool)
    turns = torch.tensor([[0, 0, -1, 1, 1], [0, 0, -1, 1, 1],
                          [0, 0, -1, -1, -1], [0, 0, -1, -1, -1]])
    adv, _ = normalized_terminal_turn_rloo(outcome, valid, turns, ["ep"] * 4)
    assert torch.allclose(adv[:, 0], torch.tensor([1.5, -0.5, -0.5, -0.5]))
    assert torch.equal(adv[:, 0], adv[:, 1])  # no division by token count
    assert torch.equal(adv[:, 2], torch.zeros(4))  # masked observation
    assert torch.equal(adv[0:2, 3], adv[0:2, 4])
    assert adv[0, 3] > 0 and adv[1, 3] < 0
    tied, _ = normalized_terminal_turn_rloo(torch.zeros(4), valid, turns, ["ep"] * 4)
    assert torch.count_nonzero(tied) == 0


if __name__ == "__main__":
    synthetic_self_check()
    print("normalized terminal RLOO synthetic checks passed")
