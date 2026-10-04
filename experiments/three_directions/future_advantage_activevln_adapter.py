"""Fail-closed mapping from live ActiveVLN response spans to sparse n=4 credit.

The frozen ranker produces two observation-only prefix scores per rollout.
This adapter does not query the ranker; it checks the rollout/token boundary
and delegates the actual group centering to future_advantage_group4_reward.
It is staged for a future two-step pilot and is not installed in verl.
"""

from __future__ import annotations

import torch

from future_advantage_group4_reward import sparse_group4_advantage


def from_activevln_sparse_batch(data, response_mask: torch.Tensor):
    required = {"learned_prefix_scores", "learned_score_valid",
                "action_mask", "task_success", "token_level_rewards"}
    if not required.issubset(data.batch):
        raise ValueError("learned sparse score or rollout tensors missing")
    scores = data.batch["learned_prefix_scores"]
    valid = data.batch["learned_score_valid"].bool()
    action_mask = data.batch["action_mask"][:, -response_mask.shape[1]:].bool()
    if scores.ndim != 2 or scores.shape[1] != 2 or \
            valid.shape != scores.shape or \
            response_mask.shape != action_mask.shape or \
            not torch.equal(action_mask, response_mask.bool()):
        raise ValueError("score tensor / generated-action mask mismatch")
    infos = data.non_tensor_batch["info"]
    group_ids = data.non_tensor_batch["uid"]
    success = data.batch["task_success"].reshape(-1).bool()
    batch_size, response_length = action_mask.shape
    if len(infos) != len(group_ids) or len(infos) != batch_size or \
            len(success) != batch_size or scores.shape[0] != batch_size:
        raise ValueError("sparse batch identity mismatch")
    turn_index = torch.full((batch_size, response_length), -1,
                            dtype=torch.long, device=action_mask.device)
    stop_mask = torch.zeros_like(action_mask)
    for row, info in enumerate(infos):
        positions = torch.nonzero(action_mask[row], as_tuple=False).flatten().tolist()
        if not positions:
            raise ValueError(f"rollout {row} has no generated action tokens")
        spans = [[positions[0]]]
        for position in positions[1:]:
            if position == spans[-1][-1] + 1:
                spans[-1].append(position)
            else:
                spans.append([position])
        turns = info["gen_traj"]
        if len(spans) != len(turns):
            raise ValueError(f"action blocks do not match executed turns: {row}")
        for column, anchor in enumerate((3, 6)):
            if bool(valid[row, column]) and len(turns) <= anchor:
                raise ValueError(f"score without post-anchor view: {row}/{anchor}")
        for turn, (detail, span) in enumerate(zip(turns, spans)):
            extracted = [str(action).strip().lower()
                         for action in detail["extracted_actions"]]
            if not extracted:
                raise ValueError(f"empty extracted action block: {row}/{turn}")
            if "stop" in extracted[:-1]:
                raise ValueError(f"motion after STOP: {row}/{turn}")
            if "stop" in extracted and turn != len(turns) - 1:
                raise ValueError(f"nonterminal STOP block: {row}/{turn}")
            turn_index[row, span] = turn
            if "stop" in extracted:
                stop_mask[row, span] = True
        for column, anchor in enumerate((3, 6)):
            if bool(valid[row, column]) and bool(stop_mask[row,
                    turn_index[row] == anchor - 1].any()):
                raise ValueError(f"scored STOP turn: {row}/{anchor}")
    outcome = data.batch["token_level_rewards"].sum(dim=-1)
    return sparse_group4_advantage(
        outcome, scores, valid, turn_index, stop_mask,
        [str(value) for value in group_ids], success)


def synthetic_self_check() -> dict:
    class Batch:
        pass

    data = Batch()
    # Eight assistant blocks with one observation token separating them.
    response = torch.zeros(4, 15, dtype=torch.bool)
    response[:, ::2] = True
    info = {"gen_traj": [{"extracted_actions": [
        "stop" if turn == 7 else "move forward 25cm"]}
        for turn in range(8)]}
    data.batch = {
        "learned_prefix_scores": torch.tensor([
            [1., 0.], [0., 1.], [0., 0.], [0., 0.]]),
        "learned_score_valid": torch.ones(4, 2, dtype=torch.bool),
        "action_mask": response.clone(),
        "task_success": torch.zeros(4, 1, dtype=torch.bool),
        "token_level_rewards": torch.zeros(4, 15),
    }
    data.non_tensor_batch = {"info": [info] * 4, "uid": ["episode"] * 4}
    advantage, _ = from_activevln_sparse_batch(data, response)
    assert torch.all(advantage[:, [0, 1, 2, 3, 6, 8, 12, 14]] == 0)
    assert advantage[0, 4] == .25 and advantage[1, 10] == .25
    data.batch["task_success"][0, 0] = True
    data.batch["token_level_rewards"][0, 14] = 15
    mixed, _ = from_activevln_sparse_batch(data, response)
    assert mixed[0, 14] > 0 and mixed[1, 14] < 0
    data.batch["task_success"][0, 0] = False
    data.batch["learned_score_valid"][3, 1] = False
    data.batch["learned_prefix_scores"][3, 1] = 0
    partial, _ = from_activevln_sparse_batch(data, response)
    assert torch.all(partial[:, 10] == 0)
    data.batch["learned_score_valid"][3, 1] = True
    altered = {"gen_traj": info["gen_traj"][:3]}
    data.non_tensor_batch["info"][3] = altered
    try:
        from_activevln_sparse_batch(data, response)
    except ValueError:
        pass
    else:
        raise AssertionError("short route with future score was accepted")
    return {"group_size": 4, "anchors": [3, 6],
            "only_scored_movement_tokens_have_auxiliary_credit": True,
            "mixed_outcome_preserved": True,
            "short_scored_route_rejected": True}


if __name__ == "__main__":
    print(synthetic_self_check())
