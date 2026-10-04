"""Recompute the frozen boundary reward on existing n=4 train rollouts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from stop_boundary_reward import boundary_turn_reward


def active_group(rows: list[list[float]]) -> bool:
    horizon = max(map(len, rows))
    returns = []
    for rewards in rows:
        running = 0.0
        values = [0.0] * len(rewards)
        for turn in range(len(rewards) - 1, -1, -1):
            running += rewards[turn]
            values[turn] = running
        returns.append(values)
    return any(len(values) >= 2 and max(values) - min(values) > 1e-7
               for turn in range(horizon)
               for values in [[row[turn] for row in returns
                               if turn < len(row)]])


def inspect(path: Path, seed: int) -> dict:
    counts = Counter()
    seen = set()
    with path.open() as stream:
        for step, line in enumerate(stream, 1):
            record = json.loads(line)
            if record["step"] != step or len(record["info"]) != 16:
                raise ValueError(f"seed {seed}: bad rollout step {step}")
            groups = defaultdict(list)
            for item in record["info"]:
                groups[str(item["episode_id"])].append(item)
            if len(groups) != 4 or set(map(len, groups.values())) != {4} or \
                    seen.intersection(groups):
                raise ValueError(f"seed {seed}: changed group-four rows")
            seen.update(groups)
            for items in groups.values():
                counts["groups"] += 1
                original, boundary = [], []
                for item in items:
                    start = float(item["oracle_start_distance"])
                    original_row, boundary_row = [], []
                    for turn in item["gen_traj"]:
                        before = float(turn["oracle_before_distance"])
                        after = float(turn["oracle_after_distance"])
                        old = float(turn["oracle_turn_progress"])
                        stop = bool(turn["oracle_stop_response"])
                        actions = turn["executed_actions"]
                        new = boundary_turn_reward(start, before, after,
                                                   actions, stop)
                        if not math.isfinite(old):
                            raise ValueError("nonfinite old oracle reward")
                        original_row.append(old)
                        boundary_row.append(new)
                        counts["turns"] += 1
                        counts["changed_turns"] += abs(new - old) > 1e-7
                        counts["inside_boundary_penalty_turns"] += \
                            before <= 3.0 and bool(actions) and not stop
                    original.append(original_row)
                    boundary.append(boundary_row)
                all_fail = not any(item["task_success"] for item in items)
                if all_fail:
                    counts["all_failure_groups"] += 1
                    counts["old_active_all_failure_groups"] += active_group(original)
                    counts["boundary_active_all_failure_groups"] += active_group(boundary)
                    counts["all_failure_groups_with_changed_turn"] += any(
                        abs(a - b) > 1e-7 for old, new in zip(original, boundary)
                        for a, b in zip(old, new))
    if step != 128 or len(seen) != 512 or counts["groups"] != 512:
        raise ValueError(f"seed {seed}: incomplete source")
    return {"seed": seed, "source": str(path), "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect(args.checkpoint_root /
                       f"oracle_turnwise_exact512_128_seed{seed}" /
                       "rollout.jsonl", seed) for seed in (11, 22, 33)]
    result = {"schema": "stop_boundary_reward_group4_signal_preflight_v1",
              "group_size": 4, "seeds": [11, 22, 33], "reports": reports,
              "interpretation": "Offline training-row reward contrast only; no policy or navigation result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
