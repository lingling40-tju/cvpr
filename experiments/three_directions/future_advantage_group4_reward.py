"""Sparse observation-score credit for exact four-rollout VLN groups.

This is an offline-tested estimator, not yet an online reward result. A
separate adapter must verify action-token spans against live ActiveVLN
rollouts before it may be used by the trainer.
"""

from __future__ import annotations

from collections import defaultdict

import torch


ANCHOR_TURNS = (2, 5)  # zero-based action turn before views 3 and 6
CLIP = .25


def sparse_group4_advantage(
    outcome_reward: torch.Tensor,
    prefix_scores: torch.Tensor,
    score_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    stop_token_mask: torch.Tensor,
    group_ids: list[str],
    success: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep normal outcome GRPO; credit failed groups only at scored turns.

    A score column is used only when all four members have an active prefix
    at that anchor. The four scores are centered and clipped separately.
    Earlier turns receive no propagated return. STOP and observation tokens
    never receive auxiliary credit.
    """
    if outcome_reward.ndim != 1 or prefix_scores.ndim != 2 or \
            prefix_scores.shape != (len(outcome_reward), 2) or \
            score_valid.shape != prefix_scores.shape or \
            action_turn_index.ndim != 2 or \
            action_turn_index.shape != stop_token_mask.shape or \
            action_turn_index.shape[0] != len(outcome_reward) or \
            success.shape != outcome_reward.shape or \
            len(group_ids) != len(outcome_reward):
        raise ValueError("incompatible group-four score or token shapes")
    if prefix_scores.device != outcome_reward.device or \
            action_turn_index.device != outcome_reward.device or \
            score_valid.device != outcome_reward.device or \
            stop_token_mask.device != outcome_reward.device or \
            success.device != outcome_reward.device:
        raise ValueError("score and token tensors must share a device")
    if not torch.isfinite(outcome_reward).all() or \
            not torch.isfinite(prefix_scores).all() or \
            not torch.all(prefix_scores[~score_valid.bool()] == 0):
        raise ValueError("nonfinite score or score in an inactive prefix")
    if (action_turn_index < -1).any() or \
            ((action_turn_index == -1) & stop_token_mask.bool()).any():
        raise ValueError("invalid action or STOP token mapping")
    groups = defaultdict(list)
    for row, eid in enumerate(group_ids):
        groups[str(eid)].append(row)
    if not groups or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("every episode group must have exactly four rollouts")
    advantages = torch.zeros_like(action_turn_index,
                                  dtype=prefix_scores.dtype)
    for members in groups.values():
        rows = torch.tensor(members, device=outcome_reward.device)
        if bool(success[rows].any()):
            values = outcome_reward[rows]
            centered = values - values.mean()
            scalar = centered / (values.std(unbiased=True) + epsilon)
            for local, row in enumerate(members):
                advantages[row, action_turn_index[row] >= 0] = scalar[local]
            continue
        for column, turn in enumerate(ANCHOR_TURNS):
            if not bool(score_valid[rows, column].all()):
                continue
            values = prefix_scores[rows, column]
            centered = (values - values.mean()).clamp(-CLIP, CLIP)
            for local, row in enumerate(members):
                mask = ((action_turn_index[row] == turn) &
                        ~stop_token_mask[row].bool())
                if not bool(mask.any()):
                    raise ValueError(f"scored prefix has no movement tokens: {row}/{turn}")
                advantages[row, mask] = centered[local]
    return advantages, advantages.clone()


def synthetic_self_check() -> dict:
    # Response positions: turn 1, observation, turn 2, turn 3, turn 4,
    # turn 5, turn 6, STOP. Only turns 3 and 6 can receive auxiliary credit.
    turns = torch.tensor([[0, -1, 1, 2, 3, 4, 5, 6]] * 4)
    stop = torch.tensor([[False] * 7 + [True]] * 4)
    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0],
                           [0.0, 0.0], [0.0, 0.0]])
    valid = torch.ones(4, 2, dtype=torch.bool)
    success = torch.zeros(4, dtype=torch.bool)
    adv, _ = sparse_group4_advantage(
        torch.zeros(4), scores, valid, turns, stop,
        ["episode"] * 4, success)
    assert torch.all(adv[:, [0, 1, 2, 4, 5, 7]] == 0)
    assert adv[0, 3] == .25 and adv[1, 6] == .25
    missing = valid.clone()
    missing[3, 1] = False
    partial_scores = scores.clone()
    partial_scores[3, 1] = 0
    partial, _ = sparse_group4_advantage(
        torch.zeros(4), partial_scores, missing, turns, stop,
        ["episode"] * 4, success)
    assert torch.all(partial[:, 6] == 0)
    mixed, _ = sparse_group4_advantage(
        torch.tensor([15., 0., 0., 0.]), scores, valid, turns, stop,
        ["episode"] * 4, torch.tensor([True, False, False, False]))
    assert mixed[0, 7] > 0 and mixed[1, 7] < 0
    try:
        sparse_group4_advantage(
            torch.zeros(4), scores, valid, turns, stop,
            ["one", "one", "two", "two"], success)
    except ValueError:
        pass
    else:
        raise AssertionError("non-four group accepted")
    return {"group_size": 4, "anchor_turns_one_based": [3, 6],
            "earlier_and_observation_tokens_zero": True,
            "all_failure_stop_zero": True,
            "partial_anchor_skipped": True,
            "mixed_outcome_preserved": True}


if __name__ == "__main__":
    print(synthetic_self_check())
