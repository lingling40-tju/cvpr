"""Independent train-row and privileged progress audit for the oracle smoke."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69"
ENV_SHA = "6d90aed3c919cf76d892aa7984b6c5586c04e07483a26310584579aa13eb1ead"
AGENT_SHA = "903c3788902a613c68d8d4c0f681eacbe3d0d4eb421c7ea4d2a45f4cde648391"
TRAINER_SHA = "2a135d65f35d0a5d9108404746ff102e69afac171b9ce6dc6764fd9c3bf10393"
HELPER_SHA = "092e7917e1756bcbbdb56ec7a58efde970c8e4c6f59a0124f7f660133ed9c9a6"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite oracle trace value")
    return result


def audit_item(item: dict, counts: Counter) -> list[float]:
    start = finite(item["oracle_start_distance"])
    turns = item["gen_traj"]
    if start < 0 or not turns:
        raise ValueError("invalid initial distance or missing turns")
    previous = start
    rewards = []
    for turn in turns:
        before = finite(turn["oracle_before_distance"])
        after = finite(turn["oracle_after_distance"])
        reward = finite(turn["oracle_turn_progress"])
        if before < 0 or after < 0 or abs(before - previous) > 1e-4:
            raise ValueError("oracle distance trace is discontinuous")
        actions = turn["executed_actions"]
        stop_generated = any(str(action).strip().lower() == "stop"
                             for action in turn["extracted_actions"])
        if bool(turn["oracle_stop_response"]) != stop_generated:
            raise ValueError("oracle STOP flag disagrees with extracted actions")
        expected = ((before - after) / max(start, 3.0)
                    if actions and not stop_generated else 0.0)
        if abs(reward - expected) > 1e-5:
            raise ValueError("oracle reward differs from distance delta")
        if stop_generated and reward != 0:
            raise ValueError("STOP received oracle movement credit")
        previous = after
        rewards.append(reward)
        counts["turns"] += 1
        counts["nonzero_progress_turns"] += abs(reward) > 1e-8
        counts["stop_turns"] += stop_generated
    if abs(previous - finite(item["distance_to_goal"])) > 1e-4 or \
            abs(rewards[-1] - finite(item["oracle_turn_progress"])) > 1e-6:
        raise ValueError("last turn disagrees with final environment info")
    components = item["reward_components"]
    expected_total = sum(finite(components[key]) for key in (
        "success_reward", "success_floor", "ndtw_reward", "semantic_reward"))
    if abs(expected_total - finite(item["total_reward"])) > 1e-3 or \
            finite(components["ndtw_reward"]) != 0 or \
            finite(components["semantic_reward"]) != 0:
        raise ValueError("destination outcome reward was altered")
    return rewards


def active_group(reward_rows: list[list[float]]) -> bool:
    """Find real turn-wise contrast among two or more active failures."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"oracle_turnwise_{args.steps}step_seed11"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control = args.control_root / "verl_checkpoints/qwen3_exact_control_64step_seed11"
    expected_files = {
        "data/qwen3_group4_exact256.parquet": DATA_SHA,
        "vlnce_server/semantic_reward/env.py": ENV_SHA,
        "verl/workers/agent/parallel_env_vlnce.py": AGENT_SHA,
        "verl/trainer/ppo/ray_trainer.py": TRAINER_SHA,
        "verl/trainer/ppo/turnwise_group4_advantage.py": HELPER_SHA,
    }
    if not (run / "completed").is_file() or not (
            checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file():
        raise ValueError("oracle train or checkpoint incomplete")
    for name, expected in expected_files.items():
        if digest(args.root / name) != expected:
            raise ValueError(f"source/data provenance mismatch: {name}")
    candidate = [json.loads(line) for line in (
        checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (
        control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != 64:
        raise ValueError("training step coverage mismatch")
    gradients = [finite(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or not all(x >= 0 for x in gradients[:args.steps]) or \
            not any(x > 1e-6 for x in gradients[:args.steps]):
        raise ValueError("no valid nonzero actor gradient")
    counts = Counter()
    seen = set()
    for step, (row, base) in enumerate(zip(candidate, baseline[:args.steps]), 1):
        if row["step"] != step or base["step"] != step or \
                len(row["info"]) != 16 or len(base["info"]) != 16:
            raise ValueError("step number or rollout count mismatch")
        groups = defaultdict(list)
        for item in row["info"]:
            groups[str(item["episode_id"])].append(item)
        base_ids = Counter(str(item["episode_id"]) for item in base["info"])
        if set(groups) != set(base_ids) or seen.intersection(groups) or \
                set(map(len, groups.values())) != {4} or set(base_ids.values()) != {4}:
            raise ValueError("candidate/control group-four rows differ")
        seen.update(groups)
        counts["groups"] += len(groups)
        for items in groups.values():
            reward_rows = [audit_item(item, counts) for item in items]
            counts["successes"] += sum(bool(item["task_success"]) for item in items)
            if not any(item["task_success"] for item in items):
                counts["all_failure_groups"] += 1
                counts["active_all_failure_groups"] += active_group(reward_rows)
    if len(seen) != 4 * args.steps or counts["groups"] != 4 * args.steps or \
            counts["nonzero_progress_turns"] == 0 or \
            counts["active_all_failure_groups"] == 0:
        raise ValueError("oracle process signal did not activate")
    report = {
        "schema": "oracle_turnwise_train_audit_v1",
        "interpretation": "Privileged train-only mechanism check; no semantic or val-unseen gain.",
        "steps": args.steps, "seed": 11, "group_size": 4,
        "counts": dict(counts),
        "nonzero_actor_gradient_steps": sum(x > 1e-6 for x in gradients[:args.steps]),
        "source_sha256": expected_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.steps == 2:
        (run / "audited").write_text("passed\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
