"""Count STOP-versus-continuation pairs in the frozen n=4 exact512 control.

This is a CPU-only train-rollout support check. It neither updates a policy
nor estimates a val-unseen gain from the proposed STOP-pair reward.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


ROLLOUT_SHA256 = "c02223c8f55d4ce6472c2893fe851d456ff062f28233a7b053b6946afa020794"
HELPER_SHA256 = "1a075580917155633959ff4d4fb572cbe01a194b4957fc7a45d1be35e6a6c89c"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--reward-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.rollout) != ROLLOUT_SHA256 or digest(args.reward_helper) != HELPER_SHA256:
        raise ValueError("frozen rollout or reward rule hash mismatch")
    spec = importlib.util.spec_from_file_location("stop_pair_group4_reward", args.reward_helper)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen reward helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with args.rollout.open() as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            infos = row["info"]
            if row["step"] != step or len(infos) != 16:
                raise ValueError("step or rollout count mismatch")
            groups = Counter(str(info["episode_id"]) for info in infos)
            if len(groups) != 4 or set(groups.values()) != {4} or seen.intersection(groups):
                raise ValueError("n=4 group or episode reuse mismatch")
            seen.update(groups)
            _, _, result = module.group_relative_adjustments(infos, 4)
            counts.update(result)
    if step != 128 or len(seen) != 512 or counts["matched_groups"] != 512:
        raise ValueError("expected exact512 control coverage missing")
    report = {
        "schema": "stop_pair_exact512_control_preflight_v1",
        "interpretation": "Train-only correlated rollout support, not reward quality or navigation gain",
        "steps": step,
        "unique_episode_groups": len(seen),
        "rollouts": 16 * step,
        "group_size": 4,
        "source_sha256": {"rollout": ROLLOUT_SHA256, "reward_helper": HELPER_SHA256},
        "counts": dict(counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
