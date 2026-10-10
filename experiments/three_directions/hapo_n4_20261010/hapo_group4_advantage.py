"""Group-conditioned horizon-adaptive policy optimization for VLN.

The estimator is a training-only adaptation of HAPO's temporal-kernel
baseline. It uses the same simulator geodesic process signal as the prior
turnwise pilot, while estimating a leave-one-trajectory-out baseline from
other rollouts of the *same* instruction. It is not a semantic reward.
"""

from __future__ import annotations

from collections import defaultdict
import os

import torch


def group4_hapo_advantage(
    outcome_reward: torch.Tensor,
    process_reward: torch.Tensor,
    turn_valid: torch.Tensor,
    action_turn_index: torch.Tensor,
    group_ids: list[str],
    *,
    sigma: float = 2.0,
    gamma: float = 0.95,
    outcome_scale: float = 15.0,
    progress_weight: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return globally normalized HAPO advantages mapped to action tokens.

    Rewards combine the existing terminal score and privileged per-turn
    simulator progress. A Gaussian kernel over turn indices estimates each
    trajectory's baseline from the other three samples in its group. Each
    rollout contributes the mean of its turn-level policy losses; within a
    turn, credit is evenly divided over that action's generated tokens.
    """
    batch_size, turn_count = process_reward.shape
    if outcome_reward.ndim != 1 or turn_valid.shape != process_reward.shape:
        raise ValueError("incompatible outcome/process/valid tensor shapes")
    if action_turn_index.ndim != 2 or action_turn_index.shape[0] != batch_size:
        raise ValueError("invalid action-turn map")
    if len(group_ids) != batch_size or outcome_reward.shape[0] != batch_size:
        raise ValueError("batch identities do not match reward tensors")
    if not (sigma > 0.0) or not (0.0 < gamma <= 1.0):
        raise ValueError("sigma and gamma must be positive")
    if outcome_scale <= 0.0 or progress_weight < 0.0 or epsilon <= 0.0:
        raise ValueError("invalid fixed reward scales")
    if not torch.isfinite(outcome_reward).all() or not torch.isfinite(process_reward).all():
        raise ValueError("nonfinite reward")
    if ((action_turn_index < -1) | (action_turn_index >= turn_count)).any():
        raise ValueError("action token maps outside turn range")

    valid = turn_valid.bool()
    groups: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        groups[str(group_id)].append(row)
    if not groups or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("HAPO pilot requires exactly four rollouts per instruction")

    # Fail closed if the token map and the environment's recorded turns differ.
    for row in range(batch_size):
        mapped = {int(v) for v in action_turn_index[row].tolist() if v >= 0}
        expected = set(torch.nonzero(valid[row], as_tuple=False).flatten().tolist())
        if mapped != expected or expected != set(range(len(expected))):
            raise ValueError(f"action-turn map/valid-turn mismatch in row {row}")

    reward = progress_weight * process_reward * valid
    reward = reward.clone()
    for row in range(batch_size):
        n_turns = int(valid[row].sum())
        if n_turns == 0:
            raise ValueError(f"rollout {row} has no executed turns")
        reward[row, n_turns - 1] += outcome_reward[row] / outcome_scale

    returns = torch.zeros_like(reward)
    running = torch.zeros(batch_size, dtype=reward.dtype, device=reward.device)
    for turn in range(turn_count - 1, -1, -1):
        running = reward[:, turn] + gamma * running
        returns[:, turn] = running

    turn_advantage = torch.zeros_like(reward)
    for rows in groups.values():
        for row in rows:
            valid_turns = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
            raw = []
            for turn in valid_turns:
                numerator = torch.zeros((), dtype=reward.dtype, device=reward.device)
                denominator = torch.zeros_like(numerator)
                for other in rows:
                    if other == row:
                        continue
                    other_turns = torch.nonzero(valid[other], as_tuple=False).flatten().tolist()
                    for other_turn in other_turns:
                        weight = torch.exp(torch.tensor(
                            -((turn - other_turn) ** 2) / (2.0 * sigma * sigma),
                            dtype=reward.dtype, device=reward.device))
                        numerator = numerator + weight * returns[other, other_turn]
                        denominator = denominator + weight
                if float(denominator) <= 0.0:
                    raise ValueError("empty temporal-kernel baseline")
                raw.append(returns[row, turn] - numerator / denominator)
            if raw:
                turn_advantage[row, torch.tensor(valid_turns, device=reward.device)] = torch.stack(raw)

    active = valid
    active_values = turn_advantage[active]
    if not torch.isfinite(active_values).all():
        raise ValueError("nonfinite HAPO advantage")
    scale = active_values.std(unbiased=False)
    if float(scale) <= epsilon:
        normalized = torch.zeros_like(turn_advantage)
    else:
        normalized = (turn_advantage - active_values.mean()) / (scale + epsilon)

    token_advantage = torch.zeros_like(action_turn_index, dtype=reward.dtype)
    for row in range(batch_size):
        n_turns = int(valid[row].sum())
        for turn in range(n_turns):
            token_mask = action_turn_index[row] == turn
            token_count = int(token_mask.sum())
            if token_count == 0:
                raise ValueError(f"turn {turn} has no action tokens in row {row}")
            token_advantage[row, token_mask] = normalized[row, turn] / (n_turns * token_count)
    if not torch.isfinite(token_advantage).all():
        raise ValueError("nonfinite token advantage")
    return token_advantage, token_advantage.clone()


def synthetic_self_check() -> dict[str, bool]:
    """Exercise mask preservation, group checks, and distinct temporal credit."""
    outcome = torch.zeros(4)
    process = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    valid = torch.tensor([[1, 1, 1]] * 4, dtype=torch.bool)
    turns = torch.tensor([[0, 0, -1, 1, 2]] * 4)
    ids = ["same"] * 4
    adv, ret = group4_hapo_advantage(outcome, process, valid, turns, ids)
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()
    assert torch.equal(adv[:, 2], torch.zeros(4))
    assert torch.all(adv.abs().sum(dim=1) > 0)
    assert not torch.allclose(adv[0, :2].sum(), adv[0, 3])
    try:
        group4_hapo_advantage(outcome, process, valid, turns, ids[:3] + ["other"])
    except ValueError:
        pass
    else:
        raise AssertionError("a split instruction group must fail closed")
    return {"finite": True, "observation_tokens_zero": True,
            "group_integrity_enforced": True, "temporal_credit_nontrivial": True}


def from_activevln_batch(data, response_mask: torch.Tensor):
    """Validate the simulator turn/reward/token mapping and apply HAPO."""
    process_tokens = data.batch["env_process_reward"]
    action_mask = data.batch["action_mask"][:, -response_mask.shape[1]:].bool()
    if process_tokens.shape != response_mask.shape or \
            not torch.equal(action_mask, response_mask.bool()) or \
            not torch.isfinite(process_tokens).all():
        raise ValueError("oracle process tensor / action mask mismatch")
    infos = data.non_tensor_batch["info"]
    group_ids = [str(x) for x in data.non_tensor_batch["uid"]]
    success = data.batch["task_success"].reshape(-1).bool()
    batch_size, response_length = process_tokens.shape
    if len(infos) != batch_size or len(group_ids) != batch_size or \
            len(success) != batch_size:
        raise ValueError("HAPO batch identity mismatch")

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
    process_reward = torch.zeros((batch_size, turn_count), dtype=process_tokens.dtype,
                                 device=process_tokens.device)
    turn_valid = torch.zeros((batch_size, turn_count), dtype=torch.bool,
                             device=process_tokens.device)
    action_turn_index = torch.full((batch_size, response_length), -1,
                                   dtype=torch.long, device=process_tokens.device)
    for row, (turns, spans) in enumerate(zip(turns_by_row, spans_by_row)):
        for turn, (info, span) in enumerate(zip(turns, spans)):
            expected = float(info["oracle_turn_progress"])
            actual = float(process_tokens[row, span].sum())
            if abs(expected - actual) > 1e-6 or \
                    any(float(process_tokens[row, pos]) != 0.0 for pos in span[:-1]):
                raise ValueError(f"oracle reward / token span mismatch row={row} turn={turn}")
            process_reward[row, turn] = expected
            turn_valid[row, turn] = True
            action_turn_index[row, span] = turn
            stop_generated = any(str(action).strip().lower() == "stop"
                                 for action in info["extracted_actions"])
            if bool(info["oracle_stop_response"]) != stop_generated:
                raise ValueError("oracle STOP flag / generated action mismatch")
            if stop_generated and expected != 0.0:
                raise ValueError("STOP response has auxiliary progress")
    if bool((process_tokens * ~action_mask).abs().sum()):
        raise ValueError("oracle progress on observation tokens")

    return group4_hapo_advantage(
        data.batch["token_level_rewards"].sum(dim=-1), process_reward,
        turn_valid, action_turn_index, group_ids,
        sigma=float(os.environ.get("VLN_HAPO_SIGMA", "2.0")),
        gamma=float(os.environ.get("VLN_HAPO_GAMMA", "0.95")),
        outcome_scale=float(os.environ.get("VLN_HAPO_OUTCOME_SCALE", "15.0")),
        progress_weight=float(os.environ.get("VLN_HAPO_PROGRESS_WEIGHT", "0.5")),
    )


if __name__ == "__main__":
    print(synthetic_self_check())
