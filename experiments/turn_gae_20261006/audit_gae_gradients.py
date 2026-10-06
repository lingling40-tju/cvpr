"""Fail-closed audit of real multimodal GAE actor and critic updates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log", type=Path)
    parser.add_argument("expected_steps", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-each-nonzero", action="store_true")
    args = parser.parse_args()
    if args.expected_steps < 1:
        raise ValueError("expected_steps must be positive")

    steps = {}
    for line in args.train_log.open(errors="replace"):
        match = re.search(r"step:(\d+) - global_seqlen", line)
        if not match:
            continue
        metrics = {}
        for name in ("actor/grad_norm", "critic/grad_norm", "critic/vf_loss"):
            found = re.search(rf"{re.escape(name)}:([-+0-9.eE]+)", line)
            if not found:
                raise ValueError(f"missing {name} at optimizer step {match.group(1)}")
            metrics[name] = float(found.group(1))
        step = int(match.group(1))
        if step in steps:
            raise ValueError(f"duplicate optimizer step {step}")
        steps[step] = metrics

    if set(steps) != set(range(1, args.expected_steps + 1)):
        raise ValueError(f"expected 1..{args.expected_steps}; observed {sorted(steps)}")
    for step, metrics in steps.items():
        for name, value in metrics.items():
            if not math.isfinite(value) or (name.endswith("grad_norm") and value < 0):
                raise ValueError(f"invalid {name}={value} at step {step}")

    actor_nonzero = [step for step, m in steps.items() if m["actor/grad_norm"] > 0]
    critic_nonzero = [step for step, m in steps.items() if m["critic/grad_norm"] > 0]
    if not actor_nonzero or not critic_nonzero:
        raise ValueError("no nonzero actor or critic gradients")
    if args.require_each_nonzero and (
        len(actor_nonzero) != args.expected_steps or len(critic_nonzero) != args.expected_steps
    ):
        raise ValueError("smoke requires a nonzero actor and critic update at every step")

    result = {
        "schema": "multimodal_gae_real_gradient_audit_v1",
        "expected_steps": args.expected_steps,
        "observed_steps": len(steps),
        "actor_nonzero_steps": len(actor_nonzero),
        "critic_nonzero_steps": len(critic_nonzero),
        "actor_zero_or_rounded_step_ids": sorted(set(steps) - set(actor_nonzero)),
        "critic_zero_or_rounded_step_ids": sorted(set(steps) - set(critic_nonzero)),
        "max_critic_value_loss": max(m["critic/vf_loss"] for m in steps.values()),
        "train_log_sha256": hashlib.sha256(args.train_log.read_bytes()).hexdigest(),
        "console_precision": "gradient norms are printed with limited precision",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
