"""Standalone, testable group-four turn-wise advantage prototype.

This module is *not yet wired into verl*. It defines the estimator that a
future two-step agent/trainer smoke must reproduce before any RL pilot.
The caller must provide explicit action-token turn indices; observation
tokens are -1 and final STOP tokens are marked separately.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def group4_turnwise_advantage(
    outcome_reward: torch.Tensor,
    process_reward: torch.Tensor,
    turn_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    stop_token_mask: torch.Tensor,
    group_ids: list[str],
    success: torch.Tensor,
    *,
    gamma: float = 1.0,
    min_reward_scale: float = 0.2,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token advantages and return targets for 4-rollout groups.

    Mixed/success groups use conventional trajectory outcome GRPO.
    All-failure groups use a within-group centered return-to-go at each
    aligned turn. STOP tokens receive no auxiliary advantage. A minimum
    reward scale prevents tiny process score margins from being amplified
    into a full-size policy update.
    """
    if outcome_reward.ndim != 1 or process_reward.ndim != 2 or \
            turn_valid.shape != process_reward.shape or \
            action_turn_index.ndim != 2 or \
            stop_token_mask.shape != action_turn_index.shape or \
            success.shape != outcome_reward.shape or \
            process_reward.shape[0] != len(outcome_reward) or \
            action_turn_index.shape[0] != len(outcome_reward) or \
            len(group_ids) != len(outcome_reward):
        raise ValueError("incompatible reward, turn, or token shapes")
    if not 0.0 < gamma <= 1.0 or min_reward_scale <= 0.0:
        raise ValueError("invalid discount or reward scale")
    if not torch.isfinite(outcome_reward).all() or \
            not torch.isfinite(process_reward).all():
        raise ValueError("nonfinite rewards")
    batch_size, turn_count = process_reward.shape
    if ((action_turn_index < -1) | (action_turn_index >= turn_count)).any():
        raise ValueError("action token maps outside turn range")
    if ((action_turn_index == -1) & stop_token_mask.bool()).any():
        raise ValueError("STOP token is not an action token")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[str(group_id)].append(index)
    if any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("every group must contain exactly four rollouts")
    valid = turn_valid.bool()
    for row in range(batch_size):
        present = {int(value) for value in action_turn_index[row].tolist() if value >= 0}
        expected = {turn for turn in range(turn_count) if valid[row, turn]}
        if present != expected:
            raise ValueError(f"action-turn map/valid turns mismatch for rollout {row}")
    advantages = torch.zeros_like(action_turn_index, dtype=process_reward.dtype)
    for indices in groups.values():
        rows = torch.tensor(indices, device=outcome_reward.device)
        if bool(success[rows].any()):
            values = outcome_reward[rows]
            centered = values - values.mean()
            scale = values.std(unbiased=True)
            scalar = centered / (scale + epsilon)
            for local, row in enumerate(indices):
                advantages[row, action_turn_index[row] >= 0] = scalar[local]
            continue
        reward = process_reward[rows] * valid[rows]
        returns_to_go = torch.zeros_like(reward)
        running = torch.zeros(4, dtype=reward.dtype, device=reward.device)
        for turn in range(turn_count - 1, -1, -1):
            running = reward[:, turn] + gamma * running
            returns_to_go[:, turn] = running
        # Use one fixed scale per group. The floor preserves the meaning of
        # reward magnitude instead of normalizing arbitrarily small noise.
        values = returns_to_go[valid[rows]]
        scale = max(float(values.std(unbiased=False)), min_reward_scale) \
            if len(values) > 1 else min_reward_scale
        for turn in range(turn_count):
            active = valid[rows, turn]
            if int(active.sum()) < 2:
                continue
            centered = returns_to_go[active, turn] - \
                returns_to_go[active, turn].mean()
            active_rows = rows[active]
            for local, row in enumerate(active_rows.tolist()):
                mask = (action_turn_index[row] == turn) & \
                    ~stop_token_mask[row].bool()
                advantages[row, mask] = centered[local] / (scale + epsilon)
    return advantages, advantages.clone()


def synthetic_self_check() -> dict:
    """Prove distinct turn credit despite equal trajectory totals."""
    outcome = torch.zeros(4)
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0],
                            [0.0, 0.0], [0.0, 0.0]])
    valid = torch.ones(4, 2, dtype=torch.bool)
    # Each row: two action tokens, one observation token, one STOP token.
    turns = torch.tensor([[0, -1, 1, 1]] * 4)
    stop = torch.tensor([[False, False, False, True]] * 4)
    advantage, _ = group4_turnwise_advantage(
        outcome, rewards, valid, turns, stop, ["episode"] * 4,
        torch.zeros(4, dtype=torch.bool))
    assert torch.allclose(advantage[0, 0], advantage[1, 0])
    assert advantage[0, 2] < advantage[1, 2]
    assert torch.equal(advantage[:, 1], torch.zeros(4))
    assert torch.equal(advantage[:, 3], torch.zeros(4))
    mixed, _ = group4_turnwise_advantage(
        torch.tensor([15.0, 0.0, 0.0, 0.0]), rewards, valid,
        turns, stop, ["episode"] * 4,
        torch.tensor([True, False, False, False]))
    assert mixed[0, 0] > 0 and mixed[1, 0] < 0
    assert mixed[0, 3] > 0  # Original outcome also credits terminal STOP.
    return {"same_total_different_second_turn": True,
            "observation_tokens_zero": True,
            "all_failure_stop_auxiliary_zero": True,
            "mixed_group_outcome_preserved": True}


if __name__ == "__main__":
    print(synthetic_self_check())
