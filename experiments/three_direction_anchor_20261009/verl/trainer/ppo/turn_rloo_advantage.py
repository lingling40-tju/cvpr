"""Turn-level return-to-go with a group-four leave-one-out baseline.

This is a training-only mechanism test. The process signal is simulator
geodesic progress, not a deployable semantic reward. The old estimator
below is retained as a synthetic reference; the adapter uses turn_rloo.
"""

from __future__ import annotations

from collections import defaultdict
import os

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


def turn_rloo_advantage(
    outcome_reward: torch.Tensor,
    process_reward: torch.Tensor,
    turn_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    stop_token_mask: torch.Tensor,
    group_ids: list[str],
    success: torch.Tensor,
    *,
    outcome_scale: float = 15.0,
    progress_weight: float = 0.5,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give each executed turn its future return minus other active rollouts.

    Terminal outcome is placed on the final generated turn, including STOP.
    Each turn has equal total policy-gradient weight, irrespective of how
    many tokens spell its action. A lone surviving rollout gets zero relative
    update because there is no comparable continuation in its group.
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
    if outcome_scale <= 0 or progress_weight < 0 or not 0 < gamma <= 1:
        raise ValueError("invalid fixed reward weights or discount")
    if not torch.isfinite(outcome_reward).all() or \
            not torch.isfinite(process_reward).all():
        raise ValueError("nonfinite reward")
    if ((action_turn_index < -1) |
            (action_turn_index >= process_reward.shape[1])).any():
        raise ValueError("action token maps outside turn range")
    valid = turn_valid.bool()
    for row in range(len(outcome_reward)):
        observed = {int(t) for t in action_turn_index[row].tolist() if t >= 0}
        expected = set(torch.nonzero(valid[row], as_tuple=False).flatten().tolist())
        if observed != expected or expected != set(range(len(expected))):
            raise ValueError("action-turn map and executed turns disagree")
    groups: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        groups[str(group_id)].append(row)
    if any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("every group must contain exactly four rollouts")
    reward = progress_weight * process_reward * valid
    reward = reward.clone()
    for row in range(len(outcome_reward)):
        last_turn = int(valid[row].sum()) - 1
        if last_turn < 0:
            raise ValueError("rollout has no executed turn")
        reward[row, last_turn] += outcome_reward[row] / outcome_scale
    returns_to_go = torch.zeros_like(reward)
    running = torch.zeros_like(outcome_reward)
    for turn in range(reward.shape[1] - 1, -1, -1):
        running = reward[:, turn] + gamma * running
        returns_to_go[:, turn] = running
    advantages = torch.zeros(action_turn_index.shape, dtype=reward.dtype,
                             device=reward.device)
    for rows in groups.values():
        for turn in range(reward.shape[1]):
            active = [row for row in rows if bool(valid[row, turn])]
            if len(active) < 2:
                continue
            values = returns_to_go[active, turn]
            centered = (len(active) * values - values.sum()) / (len(active) - 1)
            for row, value in zip(active, centered):
                mask = action_turn_index[row] == turn
                advantages[row, mask] = value / int(mask.sum())
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


def from_activevln_batch(data, response_mask: torch.Tensor):
    """Map actual response spans and simulator process rewards to turns.

    The pilot fails closed on any mismatch between the generated assistant
    blocks, the action mask, the recorded environment turns, and the
    separate privileged progress tensor. Observation tokens have index -1.
    """
    process_tokens = data.batch["env_process_reward"]
    action_mask = data.batch["action_mask"][:, -response_mask.shape[1]:].bool()
    if process_tokens.shape != response_mask.shape or \
            not torch.equal(action_mask, response_mask.bool()) or \
            not torch.isfinite(process_tokens).all():
        raise ValueError("oracle process tensor / action mask mismatch")
    infos = data.non_tensor_batch["info"]
    group_ids = data.non_tensor_batch["uid"]
    success = data.batch["task_success"].reshape(-1).bool()
    batch_size, response_length = process_tokens.shape
    if len(infos) != len(group_ids) or len(infos) != batch_size or \
            len(success) != batch_size:
        raise ValueError("oracle batch identity mismatch")
    turns_by_row = []
    spans_by_row = []
    for row in range(batch_size):
        positions = torch.nonzero(action_mask[row], as_tuple=False).flatten().tolist()
        if not positions:
            raise ValueError(f"no generated action tokens in rollout {row}")
        spans = [[positions[0]]]
        for position in positions[1:]:
            if position == spans[-1][-1] + 1:
                spans[-1].append(position)
            else:
                spans.append([position])
        turns = infos[row]["gen_traj"]
        if len(spans) != len(turns):
            raise ValueError(f"action block / executed turn mismatch in rollout {row}")
        turns_by_row.append(turns)
        spans_by_row.append(spans)
    turn_count = max(map(len, turns_by_row))
    process_reward = torch.zeros((batch_size, turn_count),
                                 dtype=process_tokens.dtype, device=process_tokens.device)
    turn_valid = torch.zeros((batch_size, turn_count), dtype=torch.bool,
                             device=process_tokens.device)
    action_turn_index = torch.full((batch_size, response_length), -1,
                                   dtype=torch.long, device=process_tokens.device)
    stop_token_mask = torch.zeros_like(action_mask)
    for row, (turns, spans) in enumerate(zip(turns_by_row, spans_by_row)):
        for turn, (info, span) in enumerate(zip(turns, spans)):
            expected = float(info["oracle_turn_progress"])
            actual = float(process_tokens[row, span].sum())
            if abs(expected - actual) > 1e-6 or \
                    any(float(process_tokens[row, pos]) != 0.0 for pos in span[:-1]):
                raise ValueError(f"oracle reward / token span mismatch in rollout {row}, turn {turn}")
            process_reward[row, turn] = expected
            turn_valid[row, turn] = True
            action_turn_index[row, span] = turn
            stop_generated = any(
                str(action).strip().lower() == "stop"
                for action in info["extracted_actions"])
            if bool(info["oracle_stop_response"]) != stop_generated:
                raise ValueError("oracle STOP flag / generated action mismatch")
            if stop_generated:
                if expected != 0:
                    raise ValueError("STOP response has auxiliary progress")
                stop_token_mask[row, span] = True
    if bool((process_tokens * ~action_mask).abs().sum()):
        raise ValueError("oracle progress on observation tokens")
    outcome_reward = data.batch["token_level_rewards"].sum(dim=-1)
    if os.environ.get("VLN_SRGPO") == "1":
        return srgpo_from_turn_batch(outcome_reward, process_reward, turn_valid,
                                     action_turn_index, group_ids)
    if os.environ.get("VLN_NORMALIZED_TERMINAL_RLOO") == "1":
        from verl.trainer.ppo.normalized_terminal_rloo import (
            normalized_terminal_turn_rloo,
        )
        return normalized_terminal_turn_rloo(
            outcome_reward, turn_valid, action_turn_index,
            [str(x) for x in group_ids],
        )
    return turn_rloo_advantage(
        outcome_reward, process_reward, turn_valid, action_turn_index,
        stop_token_mask, [str(x) for x in group_ids], success,
        gamma=float(os.environ.get("VLN_TURN_RLOO_GAMMA", "1.0")),
        progress_weight=float(os.environ.get("VLN_TURN_RLOO_PROGRESS_WEIGHT", "0.5")))


def turn_rloo_self_check() -> dict:
    outcome = torch.zeros(4)
    progress = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    valid = torch.ones(4, 2, dtype=torch.bool)
    turns = torch.tensor([[0, 0, -1, 1, 1]] * 4)
    stop = torch.zeros_like(turns, dtype=torch.bool)
    adv, _ = turn_rloo_advantage(outcome, progress, valid, turns, stop,
                                 ["episode"] * 4, torch.zeros(4, dtype=torch.bool))
    assert torch.allclose(adv[0, 0], adv[1, 0])
    assert adv[1, 3] > adv[0, 3]
    assert torch.all(adv[:, 2] == 0)
    assert torch.allclose(adv[:, 0] + adv[:, 1], adv[:, 3] + adv[:, 4]) is False
    outcome[0] = 15
    adv, _ = turn_rloo_advantage(outcome, torch.zeros_like(progress), valid,
                                 turns, stop, ["episode"] * 4,
                                 torch.tensor([True, False, False, False]))
    assert adv[0, 3] > 0 and adv[1, 3] < 0
    return {"turn_credit": True, "observation_mask": True,
            "terminal_stop_credit": True, "group_size": 4}


def srgpo_from_turn_batch(
    outcome_reward: torch.Tensor,
    process_reward: torch.Tensor,
    turn_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    group_ids: list[str],
    *,
    step_group_size: int = 16,
    step_weight: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SRGPO-style combination of trajectory GRPO and random step groups.

    Outcome advantages are normalized within each four-rollout instruction group.
    Signed, start-distance-normalized simulator progress is standardized over
    shuffled groups of 16 executed turns in the current optimizer batch. The
    resulting process signal is mapped only to tokens that generated that turn.
    This is privileged simulator feedback, not a deployable semantic verifier.
    """
    if outcome_reward.ndim != 1 or process_reward.ndim != 2 or             turn_valid.shape != process_reward.shape or action_turn_index.ndim != 2:
        raise ValueError("invalid SRGPO tensor shapes")
    if process_reward.shape[0] != len(outcome_reward) or len(group_ids) != len(outcome_reward):
        raise ValueError("SRGPO batch identities do not match tensors")
    if step_group_size < 2 or step_weight < 0 or epsilon <= 0:
        raise ValueError("invalid SRGPO grouping parameters")
    if not torch.isfinite(outcome_reward).all() or not torch.isfinite(process_reward).all():
        raise ValueError("nonfinite SRGPO signals")
    valid = turn_valid.bool()
    if not torch.isfinite(action_turn_index.float()).all():
        raise ValueError("nonfinite action-turn map")
    by_group: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        by_group[str(group_id)].append(row)
    if not by_group or any(len(rows) != 4 for rows in by_group.values()):
        raise ValueError("SRGPO requires exactly four rollouts per instruction")
    episode_adv = torch.zeros_like(outcome_reward, dtype=torch.float32)
    for rows in by_group.values():
        ids = torch.tensor(rows, device=outcome_reward.device)
        values = outcome_reward[ids].float()
        centered = values - values.mean()
        scale = values.std(unbiased=False)
        if float(scale) > epsilon:
            episode_adv[ids] = centered / (scale + epsilon)
    pairs = torch.nonzero(valid, as_tuple=False)
    step_adv = torch.zeros_like(process_reward, dtype=torch.float32)
    if len(pairs) > 0:
        order = torch.randperm(len(pairs), device=pairs.device)
        shuffled = pairs[order]
        for start in range(0, len(shuffled), step_group_size):
            block = shuffled[start:start + step_group_size]
            if len(block) < 2:
                continue
            vals = process_reward[block[:, 0], block[:, 1]].float()
            centered = vals - vals.mean()
            scale = vals.std(unbiased=False)
            if float(scale) <= epsilon:
                continue
            norm = (centered / (scale + epsilon)).clamp(-2.0, 2.0)
            step_adv[block[:, 0], block[:, 1]] = norm
    token_adv = torch.zeros_like(action_turn_index, dtype=torch.float32)
    for row in range(action_turn_index.shape[0]):
        action_tokens = action_turn_index[row] >= 0
        if bool(action_tokens.any()):
            token_adv[row, action_tokens] = episode_adv[row]
        for turn in torch.nonzero(valid[row], as_tuple=False).flatten().tolist():
            token_adv[row, action_turn_index[row] == turn] += step_weight * step_adv[row, turn]
    if not torch.isfinite(token_adv).all():
        raise ValueError("nonfinite SRGPO advantages")
    return token_adv, token_adv.clone()


def srgpo_synthetic_self_check() -> dict:
    outcome = torch.tensor([3., 0., 1., 0., 2., 0., 0., 0.])
    progress = torch.tensor([[.1, .2], [.0, -.1], [.2, .0], [-.1, .0],
                             [.1, .0], [.0, .0], [-.1, .1], [.0, -.1]])
    valid = torch.ones(8, 2, dtype=torch.bool)
    turns = torch.tensor([[0, 0, 1, 1]] * 8)
    adv, ret = srgpo_from_turn_batch(outcome, progress, valid, turns,
                                      ["a"] * 4 + ["b"] * 4)
    assert torch.equal(adv, ret) and torch.isfinite(adv).all()
    assert torch.count_nonzero(adv) > 0
    assert adv.shape == turns.shape
    obs = torch.tensor([[-1, 0, 1, -1]] * 8)
    adv, _ = srgpo_from_turn_batch(outcome, progress, valid, obs,
                                    ["a"] * 4 + ["b"] * 4)
    assert torch.equal(adv[:, 0], torch.zeros(8))
    assert torch.equal(adv[:, 3], torch.zeros(8))
    return {"n4_groups": True, "finite_nonzero_signal": True,
            "observation_tokens_zero": True, "step_group_size": 16}



if __name__ == "__main__":
    print(turn_rloo_self_check())
    print(srgpo_synthetic_self_check())
