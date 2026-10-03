"""Independent group-four train-row and STOP-pair reward audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69"
ENV_SHA = "d2420040cede386967f2203df48af16cf1dd2cac7a9732b662071eab0021eabd"
AGENT_SHA = "c5c04edbce367eea1bad8f3cdfae67cc5cbb38eb998961473bb8e9de0a8983c4"
HELPER_SHA = "1a075580917155633959ff4d4fb572cbe01a194b4957fc7a45d1be35e6a6c89c"
STOP = "stopped but goal not reached."
CAP = "number of turns exceeded."


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite reward audit value")
    return result


def audit_group(rows: list[dict], counts: Counter) -> None:
    if len(rows) != 4:
        raise ValueError("group size changed")
    if any(bool(row["task_success"]) for row in rows):
        counts["success_groups"] += 1
        expected = [0.0] * 4
        pairs = 0
    else:
        counts["all_failure_groups"] += 1
        expected = [0.0] * 4
        stops = [i for i, row in enumerate(rows) if row["end_reason"] == STOP
                 and finite(row["distance_to_goal"]) >= 3.5]
        caps = [i for i, row in enumerate(rows) if row["end_reason"] == CAP
                and finite(row["distance_to_goal"]) >= 3.0]
        counts["mixed_eligible_groups"] += bool(stops and caps)
        pairs = 0
        for stop in stops:
            for cap in caps:
                if finite(rows[stop]["distance_to_goal"]) - \
                        finite(rows[cap]["distance_to_goal"]) >= 1.0:
                    expected[stop] -= 1 / 6
                    expected[cap] += 1 / 6
                    pairs += 1
        counts["active_all_failure_groups"] += pairs > 0
        counts["qualifying_pairs"] += pairs
    if abs(sum(expected)) > 1e-7 or any(abs(x) > 0.5 + 1e-9 for x in expected):
        raise ValueError("STOP-pair credit bound violated")
    for row, vote in zip(rows, expected):
        components = row["reward_components"]
        actual = finite(components["stop_pair_ordinal"])
        if abs(actual - vote) > 1e-6 or finite(components["fused_bonus"]) != 0 or \
                finite(components["ndtw_reward"]) != 0 or \
                finite(components["semantic_reward"]) != 0:
            raise ValueError("STOP-pair reward wiring mismatch")
        if row["fused_reward"]["status"] != "disabled" or \
                abs(finite(row["stop_pair_diagnostic"]["applied_ordinal"]) - vote) > 1e-6 or \
                finite(row["stop_pair_diagnostic"]["removed_bonus"]) != 0:
            raise ValueError("teacher unexpectedly used or diagnostic mismatch")
        base = sum(finite(components[key]) for key in (
            "success_reward", "success_floor", "ndtw_reward", "semantic_reward"))
        if abs(finite(row["total_reward"]) - (base + vote)) > 1e-3:
            raise ValueError("trajectory reward total mismatch")
        counts["nonzero_aux_rollouts"] += vote != 0
        counts["successes"] += bool(row["task_success"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"stop_pair_{args.steps}step_seed11"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control = args.control_root / "verl_checkpoints/qwen3_exact_control_64step_seed11"
    sources = {
        "data/qwen3_group4_exact256.parquet": DATA_SHA,
        "vlnce_server/semantic_reward/env.py": ENV_SHA,
        "verl/workers/agent/parallel_env_vlnce.py": AGENT_SHA,
        "verl/workers/agent/stop_pair_group4_reward.py": HELPER_SHA,
    }
    if not (run / "completed").is_file() or not (
            checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file():
        raise ValueError("STOP-pair train or checkpoint incomplete")
    for relative, expected in sources.items():
        if digest(args.root / relative) != expected:
            raise ValueError(f"source/data provenance mismatch: {relative}")
    candidate = [json.loads(line) for line in (checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != 64:
        raise ValueError("training step coverage mismatch")
    gradients = [finite(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or not all(x > 1e-6 for x in gradients[:args.steps]):
        raise ValueError("missing actor gradient at a step")
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
            raise ValueError("candidate/control train-row mismatch")
        seen.update(groups)
        counts["groups"] += len(groups)
        for items in groups.values():
            audit_group(items, counts)
    if len(seen) != 4 * args.steps or counts["groups"] != 4 * args.steps or \
            counts["active_all_failure_groups"] == 0:
        raise ValueError("STOP-pair training signal did not activate")
    report = {
        "schema": "stop_pair_group4_train_audit_v1",
        "interpretation": "Privileged train-only STOP mechanism; no semantic or val-unseen gain.",
        "steps": args.steps, "seed": 11, "group_size": 4,
        "counts": dict(counts),
        "nonzero_actor_gradient_steps": args.steps,
        "source_sha256": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.steps == 2:
        (run / "audited").write_text("passed\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
