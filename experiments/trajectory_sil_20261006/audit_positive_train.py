"""Audit exact optimizer coverage, positive actor signal, and KL anchoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


KEYS = ("actor/grad_norm", "actor/kl_loss", "critic/advantages/max",
        "critic/advantages/min", "critic/score/max")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=[2, 64], required=True)
    parser.add_argument("--arm", choices=["control", "candidate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for line in args.log.open(errors="replace"):
        step = re.search(r"\bstep:(\d+) - global_seqlen", line)
        if not step:
            continue
        index = int(step.group(1))
        if index in rows:
            raise ValueError(f"duplicate optimizer step {index}")
        row = {}
        for key in KEYS:
            value = re.search(re.escape(key) + r":([-+0-9.eE]+)", line)
            if value is None:
                raise ValueError(f"missing {key} at step {index}")
            row[key] = float(value.group(1))
            if not math.isfinite(row[key]):
                raise ValueError(f"nonfinite {key} at step {index}")
        rows[index] = row
    if set(rows) != set(range(1, args.steps + 1)):
        raise ValueError("optimizer step coverage differs")
    if any(row["actor/grad_norm"] < 0 or row["critic/score/max"] < 0 or
           row["critic/score/max"] > 20.001 for row in rows.values()):
        raise ValueError("gradient or frozen terminal reward range differs")
    if args.arm == "candidate" and any(row["critic/advantages/min"] < -0.001
                                       for row in rows.values()):
        raise ValueError("positive-only candidate produced negative advantage")
    positive_steps = sum(row["critic/advantages/max"] > 0 for row in rows.values())
    nonzero_gradient_steps = sum(row["actor/grad_norm"] > 0 for row in rows.values())
    if positive_steps == 0 or nonzero_gradient_steps == 0:
        raise ValueError("no positive actor signal or no optimizer gradient")
    result = {
        "schema": "positive_trajectory_training_audit_v1",
        "arm": args.arm, "expected_steps": args.steps, "observed_steps": len(rows),
        "positive_advantage_steps": positive_steps,
        "nonzero_actor_gradient_steps": nonzero_gradient_steps,
        "zero_actor_gradient_step_ids": [step for step, row in sorted(rows.items())
                                         if row["actor/grad_norm"] == 0],
        "min_advantage": min(row["critic/advantages/min"] for row in rows.values()),
        "max_advantage": max(row["critic/advantages/max"] for row in rows.values()),
        "max_terminal_score": max(row["critic/score/max"] for row in rows.values()),
        "kl_loss_metric_present_and_finite": True,
        "train_log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "console_metric_precision": "Three decimal places; reported zeros may be rounded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"positive_advantage_steps": positive_steps,
                      "nonzero_actor_gradient_steps": nonzero_gradient_steps}))


if __name__ == "__main__":
    main()
