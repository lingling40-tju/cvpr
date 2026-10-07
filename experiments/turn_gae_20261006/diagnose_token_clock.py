"""CPU-only time-unit diagnostic on the failed GAE source; no navigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from verl.trainer.ppo.core_algos import compute_gae_advantage_return


def clock_example(tokens_per_decision: int) -> dict:
    decisions = 12
    block = tokens_per_decision + 2
    size = decisions * block
    mask = torch.zeros((4, size), dtype=torch.float64)
    for turn in range(decisions):
        mask[:, turn * block:turn * block + tokens_per_decision] = 1
    positions = torch.where(mask[0].bool())[0]
    rewards = torch.zeros_like(mask)
    rewards[0, positions[-1]] = 1
    values = torch.zeros_like(mask)
    advantages, returns = compute_gae_advantage_return(rewards, values, mask, 0.99, 0.95)
    n = len(positions)
    first_return = float(returns[0, positions[0]])
    expected = 0.99 ** (n - 1)
    if not math.isclose(first_return, expected, rel_tol=1e-12):
        raise ValueError("source recurrence differs from per-action-token discount")
    return {"environment_decisions": decisions, "generated_tokens_per_decision": tokens_per_decision,
            "action_tokens": n, "masked_observation_tokens": decisions * 2,
            "terminal_reward_first_row": 1.0, "terminal_reward_other_three_rows": 0.0,
            "critic_values": "all zero", "first_discounted_return": first_return,
            "first_unwhitened_advantage": (0.99 * 0.95) ** (n - 1),
            "first_whitened_advantage_rewarded_row": float(advantages[0, positions[0]]),
            "last_whitened_advantage_rewarded_row": float(advantages[0, positions[-1]]),
            "negative_action_tokens_in_rewarded_row": int((advantages[0, positions] < 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError("diagnostic must run without a CUDA device")
    core = args.root / "verl/trainer/ppo/core_algos.py"
    metrics = args.root / "verl/trainer/ppo/metric_utils.py"
    log = args.root / "runlogs/turn_gae_64step_seed11/train.log"
    rows = []
    for line in log.open(errors="replace"):
        step = re.search(r"step:(\d+) - global_seqlen", line)
        if not step:
            continue
        row = {"step": int(step.group(1))}
        for name, value in re.findall(r"(?:^| )([a-zA-Z_0-9/]+):([-+0-9.eE]+)", line):
            if any(name.startswith(prefix) for prefix in ("response_length/", "critic/values/",
                                                          "critic/vf_explained_var", "critic/score/")):
                row[name] = float(value)
        rows.append(row)
    if [row["step"] for row in rows] != list(range(1, 65)):
        raise ValueError("failed GAE run lacks exact 64-step coverage")
    token_clock = clock_example(20)
    decision_clock = clock_example(1)
    if token_clock["first_whitened_advantage_rewarded_row"] >= 0 or \
            decision_clock["first_whitened_advantage_rewarded_row"] <= 0:
        raise ValueError("synthetic length/whitening contrast not reproduced")
    result = {"schema": "gae_token_clock_cpu_diagnostic_v1", "cuda_visible": False,
              "gamma": 0.99, "lambda": 0.95,
              "source_sha256": {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (core, metrics, log)},
              "logged_optimizer_steps": len(rows),
              "logged_first_last_step": [rows[0], rows[-1]],
              "logged_mean_action_tokens_min_max": [min(row["response_length/mean"] for row in rows),
                                                    max(row["response_length/mean"] for row in rows)],
              "synthetic_token_clock": token_clock,
              "synthetic_decision_clock": decision_clock,
              "interpretation": "The frozen failed GAE path discounts generated action-text tokens, while skipping observation spans. The synthetic zero-value example changes early credit solely with text length. This does not identify the cause of the measured navigation decline, evaluate a corrected model, or authorize retuning the failed development screen."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
