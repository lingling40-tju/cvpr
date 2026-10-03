"""Independent row, reward, teacher, and gradient audit for confidence pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


DATA_SHA = "6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69"
HELPER_SHA = "ad327090b281b3e6dd103badcf7173a86ead18ffd904f6bf6c85c097249690d9"
AGENT_SHA = "c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256"
MODES = ("stopped but goal not reached.", "number of turns exceeded.")
GAP = 5.5


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_votes(items: list[dict]) -> tuple[list[float], int]:
    """Compute audit-only votes without importing the trainer reward module."""
    votes = [0.0] * len(items)
    confident_pairs = 0
    for mode in MODES:
        positions = [i for i, item in enumerate(items)
                     if item["end_reason"] == mode]
        if len(positions) < 2:
            continue
        for left, right in itertools.combinations(positions, 2):
            a = float(items[left]["fused_reward"]["raw"])
            b = float(items[right]["fused_reward"]["raw"])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("nonfinite audit teacher margin")
            if abs(a - b) < GAP:
                continue
            direction = 1 if a > b else -1
            delta = direction / (2 * (len(positions) - 1))
            votes[left] += delta
            votes[right] -= delta
            confident_pairs += 1
        if abs(sum(votes[i] for i in positions)) > 1e-6:
            raise ValueError("audit reward is not zero-sum")
    return votes, confident_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"qwen_confident_{args.steps}step_seed11"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control = (args.source_root / "verl_checkpoints" /
               "qwen3_exact_control_64step_seed11")
    if not (run / "completed").is_file() or not (
            checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file() or \
            digest(args.root / "data/qwen3_group4_exact256.parquet") != DATA_SHA or \
            digest(args.root / "verl/workers/agent/qwen_group_reward.py") != HELPER_SHA or \
            digest(args.root / "verl/workers/agent/parallel_env_vlnce.py") != AGENT_SHA:
        raise ValueError("incomplete confident run or source provenance mismatch")
    candidate = [json.loads(line) for line in (
        checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (
        control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != 64:
        raise ValueError("training step coverage mismatch")
    before = json.loads((run / "reward_health_before.json").read_text())
    after = json.loads((run / "reward_health_after.json").read_text())
    if before["variant"] != "qwen3_exact_start_group_rank_v1" or \
            after["variant"] != "qwen3_exact_start_group_rank_v1" or \
            before["manifest_sha256"] != \
            "1effbefb1ac9f55edf7470f9d9c83c2b4875fea71daa7199786f99a4c5ea76e1" or \
            before["manifest_sha256"] != after["manifest_sha256"] or \
            before["expert_analysis_sha256"] != \
            "6e72f6b2a8c1b73216a1f01c70cb85fc733e592035799e3a4bbf12774edca1d4" or \
            before["expert_analysis_sha256"] != after["expert_analysis_sha256"]:
        raise ValueError("teacher service changed during training")
    gradients = [float(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or not all(
            math.isfinite(x) and x >= 0 for x in gradients[:args.steps]) or \
            not any(x > 1e-6 for x in gradients[:args.steps]):
        raise ValueError("optimizer gradient audit failed")
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
                set(map(len, groups.values())) != {4} or \
                set(base_ids.values()) != {4}:
            raise ValueError("control/candidate train-row pairing mismatch")
        seen.update(groups)
        counts["groups"] += len(groups)
        for items in groups.values():
            counts["diverse_groups"] += len({tuple(
                turn["response"] for turn in item["gen_traj"]) for item in items}) > 1
            success = any(item["task_success"] for item in items)
            counts["all_failure_groups"] += not success
            for item in items:
                components = item["reward_components"]
                teacher = item["fused_reward"]
                ordinal = float(components["qwen_group_ordinal"])
                if not math.isfinite(ordinal) or abs(ordinal) > 0.5 + 1e-6 or \
                        float(components["fused_bonus"]) != 0 or \
                        float(components["ndtw_reward"]) != 0 or \
                        float(components["semantic_reward"]) != 0 or \
                        float(teacher["removed_bonus"]) != 0 or \
                        abs(ordinal - float(teacher["applied_ordinal"])) > 1e-6:
                    raise ValueError("invalid candidate group reward wiring")
                if item["task_success"]:
                    counts["successes"] += 1
                    if teacher["status"] != "disabled" or ordinal != 0:
                        raise ValueError("success was assigned semantic rank")
                else:
                    counts["failures_scored"] += 1
                    if teacher["status"] != "ok" or teacher["scored_views"] != 6:
                        raise ValueError("failed rollout lacks teacher score")
                expected_total = sum(float(components[key]) for key in (
                    "success_reward", "success_floor", "ndtw_reward",
                    "semantic_reward", "qwen_group_ordinal"))
                if abs(float(item["total_reward"]) - expected_total) > 1e-3:
                    raise ValueError("total reward does not match components")
                counts["nonzero_ordinal_rollouts"] += ordinal != 0
            if success:
                if any(float(item["reward_components"]["qwen_group_ordinal"])
                       for item in items):
                    raise ValueError("mixed group got semantic rank")
                continue
            expected, confident_pairs = expected_votes(items)
            counts["confident_pairs"] += confident_pairs
            counts["ordinal_active_groups"] += any(value != 0 for value in expected)
            for item, value in zip(items, expected):
                if abs(float(item["reward_components"]["qwen_group_ordinal"]) - value) > 1e-6:
                    raise ValueError("confidence-pair reward disagrees with audit votes")
    if len(seen) != args.steps * 4 or counts["groups"] != args.steps * 4 or \
            counts["successes"] + counts["failures_scored"] != args.steps * 16 or \
            after["requests"] - before["requests"] != counts["failures_scored"] or \
            counts["diverse_groups"] == 0 or counts["ordinal_active_groups"] == 0:
        raise ValueError("training outcome or teacher request coverage mismatch")
    report = {
        "schema": "qwen_confident_pair_train_audit_v1",
        "interpretation": "Matched train-row and reward wiring only; no val-unseen navigation claim.",
        "steps": args.steps, "seed": 11, "group_size": 4,
        "min_score_gap": GAP,
        "counts": dict(counts),
        "reward_requests": after["requests"] - before["requests"],
        "nonzero_actor_gradient_steps": sum(x > 1e-6 for x in gradients[:args.steps]),
        "helper_sha256": HELPER_SHA,
        "agent_sha256": AGENT_SHA,
        "manifest_sha256": after["manifest_sha256"],
        "expert_analysis_sha256": after["expert_analysis_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
