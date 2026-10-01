"""Fail if a short GRPO run had duplicate group samples or zero actor updates."""

import argparse
import collections
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("train_log", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.rollout.read_text().splitlines()]
    assert [record["step"] for record in records] == list(range(1, args.steps + 1))
    groups = diverse = varied_return = 0
    for record in records:
        by_episode = collections.defaultdict(list)
        for item in record["info"]:
            by_episode[str(item["episode_id"])].append(item)
        for pair in by_episode.values():
            assert len(pair) == 2
            groups += 1
            signatures = [tuple(turn["response"] for turn in item["gen_traj"])
                          for item in pair]
            diverse += signatures[0] != signatures[1]
            varied_return += pair[0]["total_reward"] != pair[1]["total_reward"]
    grad_norms = [float(value) for value in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", args.train_log.read_text())]
    result = {
        "steps": len(records),
        "groups": groups,
        "diverse_trajectory_groups": diverse,
        "nonzero_return_variance_groups": varied_return,
        "actor_grad_norms": grad_norms,
    }
    print(json.dumps(result, indent=2))
    assert diverse > 0, "all GRPO groups had identical trajectories"
    assert varied_return > 0, "all GRPO groups had identical returns"
    assert any(value > 1e-6 for value in grad_norms), "no nonzero actor gradient"


if __name__ == "__main__":
    main()
