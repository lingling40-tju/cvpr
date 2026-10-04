"""Audit each paired 128-step, n=4 oracle/control scale run.

Recomputes train-only privileged turn rewards and checks identical episode
rows per optimizer step. This is a mechanism audit, not navigation evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from pathlib import Path
import re

from audit_oracle_turnwise_train import active_group, audit_item, digest, finite


DATA_SHA = "d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f"
SOURCE_SHA = {
    "vlnce_server/semantic_reward/env.py": "6d90aed3c919cf76d892aa7984b6c5586c04e07483a26310584579aa13eb1ead",
    "verl/workers/agent/parallel_env_vlnce.py": "903c3788902a613c68d8d4c0f681eacbe3d0d4eb421c7ea4d2a45f4cde648391",
    "verl/trainer/ppo/ray_trainer.py": "2a135d65f35d0a5d9108404746ff102e69afac171b9ce6dc6764fd9c3bf10393",
    "verl/trainer/ppo/turnwise_group4_advantage.py": "092e7917e1756bcbbdb56ec7a58efde970c8e4c6f59a0124f7f660133ed9c9a6",
}


def gradient_steps(log: Path, *, every_step: bool) -> int:
    norms = [finite(value) for value in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", log.read_text())]
    if len(norms) < 128 or any(value < 0 for value in norms[:128]):
        raise ValueError(f"missing or invalid actor gradients: {log}")
    positive = sum(value > 1e-6 for value in norms[:128])
    # A destination-only control can have legitimate all-failure batches
    # with zero advantage. The turn-wise oracle must still activate every
    # step, as its 64-step wiring audit did.
    if (every_step and positive != 128) or (not every_step and positive == 0):
        raise ValueError(f"invalid nonzero-gradient coverage: {positive}/128")
    return positive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_name = f"oracle_turnwise_exact512_128_seed{args.seed}"
    control_name = f"oracle_exact512_control_128_seed{args.seed}"
    candidate_run = args.candidate_root / "runlogs" / candidate_name
    control_run = args.control_root / "runlogs" / control_name
    candidate_checkpoint = args.candidate_root / "verl_checkpoints" / candidate_name
    control_checkpoint = args.control_root / "verl_checkpoints" / control_name
    for root, run, checkpoint in (
            (args.candidate_root, candidate_run, candidate_checkpoint),
            (args.control_root, control_run, control_checkpoint)):
        if not (run / "completed").is_file() or not (
                checkpoint / "global_step_128/actor/huggingface/config.json").is_file():
            raise ValueError(f"training or checkpoint incomplete: {run}")
        if digest(root / "data/qwen3_group4_exact512.parquet") != DATA_SHA:
            raise ValueError(f"scaled train rows changed: {root}")
    for name, expected in SOURCE_SHA.items():
        if digest(args.candidate_root / name) != expected:
            raise ValueError(f"candidate source changed: {name}")
    candidate_gradients = gradient_steps(candidate_run / "train.log",
                                         every_step=True)
    control_gradients = gradient_steps(control_run / "train.log",
                                       every_step=False)
    counts = Counter()
    seen = set()
    with (candidate_checkpoint / "rollout.jsonl").open() as candidate_stream, \
            (control_checkpoint / "rollout.jsonl").open() as control_stream:
        for step, (candidate_line, control_line) in enumerate(
                zip_longest(candidate_stream, control_stream), 1):
            if candidate_line is None or control_line is None or step > 128:
                raise ValueError("candidate/control rollout step count mismatch")
            candidate = json.loads(candidate_line)
            control = json.loads(control_line)
            if candidate["step"] != step or control["step"] != step or \
                    len(candidate["info"]) != 16 or len(control["info"]) != 16:
                raise ValueError(f"step or rollout count mismatch at {step}")
            groups = defaultdict(list)
            for item in candidate["info"]:
                groups[str(item["episode_id"])].append(item)
            baseline = Counter(str(item["episode_id"]) for item in control["info"])
            if set(groups) != set(baseline) or len(groups) != 4 or \
                    set(map(len, groups.values())) != {4} or \
                    set(baseline.values()) != {4} or seen.intersection(groups):
                raise ValueError(f"same-row n4 pairing failed at {step}")
            seen.update(groups)
            counts["groups"] += 4
            for items in groups.values():
                rewards = [audit_item(item, counts) for item in items]
                if not any(item["task_success"] for item in items):
                    counts["all_failure_groups"] += 1
                    counts["active_all_failure_groups"] += active_group(rewards)
            counts["candidate_successes"] += sum(bool(item["task_success"])
                                                for item in candidate["info"])
            counts["control_successes"] += sum(bool(item["task_success"])
                                              for item in control["info"])
    if len(seen) != 512 or counts["groups"] != 512 or \
            counts["nonzero_progress_turns"] == 0 or \
            counts["active_all_failure_groups"] == 0:
        raise ValueError("scaled group or process-signal coverage failed")
    report = {
        "schema": "oracle_turnwise_exact512_scale_train_audit_v1",
        "seed": args.seed, "steps": 128, "group_size": 4,
        "unique_train_episodes": len(seen),
        "same_row_candidate_control": True,
        "nonzero_actor_gradient_steps": {"candidate": candidate_gradients,
                                         "control": control_gradients},
        "counts": dict(counts),
        "source_sha256": {"train_parquet": DATA_SHA, **SOURCE_SHA},
        "interpretation": (
            "Training-only privileged geodesic mechanism audit; no "
            "observation-only reward or navigation gain is implied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
