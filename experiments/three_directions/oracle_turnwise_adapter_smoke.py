"""Synthetic fail-closed checks for the ActiveVLN oracle-turn adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from verl.trainer.ppo.turnwise_group4_advantage import from_activevln_batch


def make_batch():
    action_mask = torch.tensor([[1, 1, 0, 1, 1]] * 4, dtype=torch.long)
    process = torch.tensor([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    infos = []
    for first, second in ((1., 0.), (0., 1.), (0., 0.), (0., 0.)):
        infos.append({"gen_traj": [
            {"oracle_turn_progress": first, "executed_actions": ["move forward 25cm"],
             "extracted_actions": ["move forward 25cm"], "oracle_stop_response": False},
            {"oracle_turn_progress": second, "executed_actions": ["turn left 15 degrees"],
             "extracted_actions": ["turn left 15 degrees"], "oracle_stop_response": False},
        ]})
    return SimpleNamespace(
        batch={
            "action_mask": action_mask,
            "env_process_reward": process,
            "task_success": torch.zeros((4, 1), dtype=torch.bool),
            "token_level_rewards": torch.zeros((4, 5)),
        },
        non_tensor_batch={"info": infos, "uid": ["episode"] * 4},
    )


def main():
    data = make_batch()
    mask = data.batch["action_mask"].bool()
    advantage, returns = from_activevln_batch(data, mask)
    assert torch.equal(advantage, returns)
    assert torch.allclose(advantage[0, :2], advantage[1, :2])
    assert torch.all(advantage[0, 3:] < advantage[1, 3:])
    assert torch.all(advantage[:, 2] == 0)
    bad = make_batch()
    bad.non_tensor_batch["info"][0]["gen_traj"][0]["oracle_turn_progress"] = 0.5
    try:
        from_activevln_batch(bad, mask)
    except ValueError as exc:
        assert "oracle reward / token span mismatch" in str(exc)
    else:
        raise AssertionError("mismatched process reward was accepted")
    stop = make_batch()
    final = stop.non_tensor_batch["info"][0]["gen_traj"][1]
    final["executed_actions"] = []  # budget exhausted before STOP executes
    final["extracted_actions"] = ["stop"]
    final["oracle_stop_response"] = True
    stopped_advantage, _ = from_activevln_batch(stop, mask)
    assert torch.all(stopped_advantage[0, 3:] == 0)
    final["oracle_stop_response"] = False
    try:
        from_activevln_batch(stop, mask)
    except ValueError as exc:
        assert "STOP flag / generated action mismatch" in str(exc)
    else:
        raise AssertionError("mismatched STOP flag was accepted")
    print("oracle turnwise adapter smoke passed")


if __name__ == "__main__":
    main()
