"""Audit exact TensorBoard updates in zero-terminal-score training batches.

Use after all 64 steps complete. The terminal score scalar comes from
`token_level_scores`, whereas the actor advantage is the selected RL
estimator. A zero terminal score with nonzero advantage is evidence of
an auxiliary training signal, subject to the frozen source/config audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = (
    "actor/grad_norm",
    "critic/score/max",
    "critic/advantages/max",
    "critic/advantages/min",
)


def read_arm(path: Path, steps: int) -> dict:
    acc = EventAccumulator(str(path))
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    if not set(TAGS) <= available:
        raise ValueError(f"missing scalar tags in {path}: {set(TAGS)-available}")
    series = {}
    for tag in TAGS:
        events = acc.Scalars(tag)
        if len(events) != len({int(e.step) for e in events}):
            raise ValueError(f"duplicate scalar steps for {tag}: {path}")
        series[tag] = {int(e.step): float(e.value) for e in events}
    expected = set(range(1, steps + 1))
    if any(set(values) != expected for values in series.values()):
        raise ValueError(f"incomplete or duplicate step coverage: {path}")
    if any(not math.isfinite(value) for values in series.values() for value in values.values()):
        raise ValueError(f"nonfinite scalar: {path}")
    zero_score = [i for i in sorted(expected) if series["critic/score/max"][i] == 0]
    zero_grad = [i for i in sorted(expected) if series["actor/grad_norm"][i] == 0]
    contrasting = [
        i for i in zero_score
        if (series["critic/advantages/max"][i] > 0 or
            series["critic/advantages/min"][i] < 0)
        and series["actor/grad_norm"][i] > 0
    ]
    events = sorted(path.glob("events.out*"))
    return {
        "steps": steps,
        "zero_terminal_score_steps": zero_score,
        "zero_actor_gradient_steps": zero_grad,
        "zero_terminal_score_with_nonzero_advantage_and_gradient": contrasting,
        "nonzero_actor_gradient_steps": steps - len(zero_grad),
        "mean_actor_grad_norm": sum(series["actor/grad_norm"].values()) / steps,
        "event_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in events},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control_events", type=Path)
    ap.add_argument("terminal_rloo_events", type=Path)
    ap.add_argument("dense_grpo_events", type=Path)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.steps != 64:
        raise ValueError("the frozen comparison has exactly 64 steps")
    rows = {
        "control": read_arm(args.control_events, args.steps),
        "terminal_rloo": read_arm(args.terminal_rloo_events, args.steps),
        "dense_grpo": read_arm(args.dense_grpo_events, args.steps),
    }
    out = {
        "schema": "factorial_zero_terminal_score_signal_audit_v1",
        "training_seed": 11,
        "group_size": 4,
        "arms": rows,
        "interpretation": "Training-batch diagnostic only, not navigation performance or a causal ablation by itself.",
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
