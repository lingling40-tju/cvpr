"""Fail-closed n=4 training audit for the boundary-aware privileged pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from stop_boundary_reward import boundary_turn_reward


DATA_SHA = "6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69"
SOURCE_SHA = {
    "vlnce_server/semantic_reward/env.py": "63a7ec72c5fef68fed7e8a7ea384a829fecf5d9665673151b1c6314009182661",
    "vlnce_server/semantic_reward/stop_boundary_reward.py": "8f3e37b89e4e903d0e10a7ec14ae64f784bb76835b93e310865b2cd75572b0b1",
    "verl/workers/agent/parallel_env_vlnce.py": "903c3788902a613c68d8d4c0f681eacbe3d0d4eb421c7ea4d2a45f4cde648391",
    "verl/trainer/ppo/ray_trainer.py": "2a135d65f35d0a5d9108404746ff102e69afac171b9ce6dc6764fd9c3bf10393",
    "verl/trainer/ppo/turnwise_group4_advantage.py": "092e7917e1756bcbbdb56ec7a58efde970c8e4c6f59a0124f7f660133ed9c9a6",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_group(reward_rows: list[list[float]]) -> bool:
    horizon = max(map(len, reward_rows))
    returns = []
    for rewards in reward_rows:
        running = 0.0
        values = [0.0] * len(rewards)
        for turn in range(len(rewards) - 1, -1, -1):
            running += rewards[turn]
            values[turn] = running
        returns.append(values)
    for turn in range(horizon):
        values = [row[turn] for row in returns if turn < len(row)]
        if len(values) >= 2 and max(values) - min(values) > 1e-7:
            return True
    return False


def audit_item(item: dict, counts: Counter) -> list[float]:
    start = float(item["oracle_start_distance"])
    previous = start
    rewards = []
    for turn in item["gen_traj"]:
        before = float(turn["oracle_before_distance"])
        after = float(turn["oracle_after_distance"])
        actual = float(turn["oracle_turn_progress"])
        stop = any(str(action).strip().lower() == "stop"
                   for action in turn["extracted_actions"])
        if not all(math.isfinite(v) and v >= 0 for v in (start, before, after)) or \
                not math.isfinite(actual) or abs(before - previous) > 1e-4 or \
                bool(turn["oracle_stop_response"]) != stop:
            raise ValueError("discontinuous or invalid boundary turn")
        expected = boundary_turn_reward(
            start, before, after, turn["executed_actions"], stop)
        if abs(actual - expected) > 1e-6 or (stop and actual != 0):
            raise ValueError("boundary process reward mismatch")
        counts["turns"] += 1
        counts["inside_penalty_turns"] += bool(
            before <= 3 and turn["executed_actions"] and not stop)
        counts["nonzero_process_turns"] += abs(actual) > 1e-7
        previous = after
        rewards.append(actual)
    if not rewards or abs(previous - float(item["distance_to_goal"])) > 1e-4 or \
            abs(rewards[-1] - float(item["oracle_turn_progress"])) > 1e-6:
        raise ValueError("missing or inconsistent boundary trajectory")
    components = item["reward_components"]
    outcome = sum(float(components[key]) for key in
                  ("success_reward", "success_floor", "ndtw_reward",
                   "semantic_reward"))
    if not math.isfinite(outcome) or \
            abs(outcome - float(item["total_reward"])) > 1e-3 or \
            float(components["ndtw_reward"]) != 0 or \
            float(components["semantic_reward"]) != 0:
        raise ValueError("outcome reward changed")
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"stop_boundary_{args.steps}step_seed11"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control = (args.control_root / "verl_checkpoints" /
               "qwen3_exact_control_64step_seed11")
    if not (run / "completed").is_file() or \
            not (checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file() or \
            digest(args.root / "data/qwen3_group4_exact256.parquet") != DATA_SHA:
        raise ValueError("checkpoint, data, or training incomplete")
    for relative, expected in SOURCE_SHA.items():
        if digest(args.root / relative) != expected:
            raise ValueError(f"changed boundary source: {relative}")
    gradients = [float(value) for value in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or \
            any(not math.isfinite(value) or value <= 1e-6
                for value in gradients[:args.steps]):
        raise ValueError("missing or zero actor gradients")
    counts = Counter()
    seen = set()
    step = 0
    with (checkpoint / "rollout.jsonl").open() as candidate_stream, \
            (control / "rollout.jsonl").open() as control_stream:
        for step, candidate_line in enumerate(candidate_stream, 1):
            if step > args.steps:
                raise ValueError("too many candidate rollout steps")
            control_line = next(control_stream)
            candidate = json.loads(candidate_line)
            baseline = json.loads(control_line)
            if candidate["step"] != step or baseline["step"] != step or \
                    len(candidate["info"]) != 16 or len(baseline["info"]) != 16:
                raise ValueError("step or rollout count mismatch")
            groups = defaultdict(list)
            for item in candidate["info"]:
                groups[str(item["episode_id"])].append(item)
            baseline_counts = Counter(str(item["episode_id"])
                                      for item in baseline["info"])
            if len(groups) != 4 or set(map(len, groups.values())) != {4} or \
                    set(groups) != set(baseline_counts) or \
                    set(baseline_counts.values()) != {4} or seen.intersection(groups):
                raise ValueError("mismatched n4 training episodes")
            seen.update(groups)
            counts["groups"] += 4
            for items in groups.values():
                reward_rows = [audit_item(item, counts) for item in items]
                counts["successes"] += sum(bool(item["task_success"])
                                           for item in items)
                if not any(item["task_success"] for item in items):
                    counts["all_failure_groups"] += 1
                    counts["active_all_failure_groups"] += active_group(reward_rows)
    if step != args.steps or len(seen) != 4 * args.steps or \
            counts["groups"] != 4 * args.steps or \
            counts["active_all_failure_groups"] == 0 or \
            counts["nonzero_process_turns"] == 0:
        raise ValueError("boundary training signal absent or incomplete")
    if args.steps == 64 and counts["inside_penalty_turns"] == 0:
        raise ValueError("no observed inside-boundary continuation penalty")
    report = {
        "schema": "stop_boundary_group4_train_audit_v1",
        "steps": args.steps, "seed": 11, "group_size": 4,
        "same_row_candidate_control": True,
        "unique_train_episodes": len(seen),
        "nonzero_actor_gradient_steps": args.steps,
        "counts": dict(counts),
        "source_sha256": {"train_parquet": DATA_SHA, **SOURCE_SHA},
        "interpretation": "Privileged n4 mechanism audit; no learned reward or navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.steps == 2:
        (run / "audited").write_text("passed\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
