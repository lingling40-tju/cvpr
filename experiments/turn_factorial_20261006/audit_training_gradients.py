"""Audit optimizer-step coverage and report sparse actor-gradient updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_log", type=Path)
    ap.add_argument("expected_steps", type=int)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    steps = {}
    for line in args.train_log.open(errors="replace"):
        match = re.search(r"step:(\d+) - global_seqlen", line)
        if match:
            grad = re.search(r"actor/grad_norm:([-+0-9.eE]+)", line)
            if not grad:
                raise ValueError("missing actor gradient in " + str(match.group(1)))
            steps[int(match.group(1))] = float(grad.group(1))
    assert set(steps) == set(range(1, args.expected_steps + 1))
    if not all(math.isfinite(x) and x >= 0 for x in steps.values()):
        raise ValueError("nonfinite or negative actor gradient")
    nonzero = sum(x > 0 for x in steps.values())
    if nonzero == 0:
        raise ValueError("all actor gradients are zero")
    result = {
        "expected_steps": args.expected_steps,
        "observed_steps": len(steps),
        "nonzero_gradient_steps": nonzero,
        "zero_or_rounded_gradient_step_ids": [step for step, value in sorted(steps.items()) if value == 0],
        "gradient_log_precision": "console values rounded to three decimal places",
        "min_actor_grad_norm": min(steps.values()),
        "max_actor_grad_norm": max(steps.values()),
        "train_log_sha256": hashlib.sha256(args.train_log.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
if __name__ == "__main__":
    main()
