"""Measure how often the matched n=4 outcome controls have no group signal.

This reads complete R2R-train rollouts for three control seeds. It does not
inspect candidate results or val-unseen data and cannot establish navigation
gain. Repeated episode IDs across seeds are kept separate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


DATA_SHA = "d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def one_seed(root: Path, seed: int) -> tuple[dict, set[str]]:
    label = f"oracle_exact512_control_128_seed{seed}"
    run = root / "runlogs" / label
    checkpoint = root / "verl_checkpoints" / label
    rollout = checkpoint / "rollout.jsonl"
    if not (run / "completed").is_file() or \
            not (checkpoint / "global_step_128/actor/huggingface/config.json").is_file():
        raise ValueError(f"incomplete control: {label}")
    counts = Counter()
    seen = set()
    step_groups = []
    with rollout.open() as stream:
        for step, line in enumerate(stream, 1):
            batch = json.loads(line)
            if batch["step"] != step or len(batch["info"]) != 16:
                raise ValueError(f"invalid n=4 batch: {label} step {step}")
            groups = defaultdict(list)
            for item in batch["info"]:
                groups[str(item["episode_id"])].append(item)
            if len(groups) != 4 or any(len(items) != 4 for items in groups.values()) or \
                    seen.intersection(groups):
                raise ValueError(f"invalid episode grouping: {label} step {step}")
            seen.update(groups)
            step_groups.append(tuple(sorted(groups)))
            for items in groups.values():
                counts["episode_groups"] += 1
                successes = sum(bool(item["task_success"]) for item in items)
                counts[f"successes_{successes}_of_4"] += 1
                counts["successful_rollouts"] += successes
                rewards = [float(item["total_reward"]) for item in items]
                if not all(math.isfinite(r) for r in rewards):
                    raise ValueError("nonfinite outcome reward")
                if successes == 0:
                    counts["all_failure_groups"] += 1
                    if all(r == 0 for r in rewards):
                        counts["all_failure_zero_reward_groups"] += 1
                    same_mode = 0
                    separated = 0
                    for left, right in itertools.combinations(items, 2):
                        if left["end_reason"] != right["end_reason"]:
                            continue
                        same_mode += 1
                        ldist = float(left["distance_to_goal"])
                        rdist = float(right["distance_to_goal"])
                        if not math.isfinite(ldist + rdist):
                            raise ValueError("nonfinite failure distance")
                        separated += abs(ldist - rdist) >= 1.0
                    counts["same_mode_failure_pairs"] += same_mode
                    counts["same_mode_gap1m_failure_pairs"] += separated
                    counts["all_failure_groups_with_same_mode_gap1m"] += separated > 0
                if len(set(rewards)) == 1:
                    counts["tied_terminal_reward_groups"] += 1
                else:
                    counts["nonzero_terminal_reward_variance_groups"] += 1
                if len({tuple(turn["response"] for turn in item["gen_traj"])
                        for item in items}) > 1:
                    counts["text_diverse_groups"] += 1
    if len(step_groups) != 128 or len(seen) != 512 or \
            counts["episode_groups"] != 512 or \
            counts["all_failure_groups"] != counts["all_failure_zero_reward_groups"]:
        raise ValueError(f"control coverage or reward semantics changed: {label}")
    report = {"seed": seed, "steps": 128, "group_size": 4,
              "unique_train_episodes": len(seen),
              "source_rollout_sha256": digest(rollout),
              "counts": dict(counts),
              "all_failure_group_fraction": counts["all_failure_groups"] / 512,
              "all_failure_same_mode_gap1m_group_fraction":
                  counts["all_failure_groups_with_same_mode_gap1m"] /
                  counts["all_failure_groups"],
              "nonzero_terminal_variance_fraction":
                  counts["nonzero_terminal_reward_variance_groups"] / 512,
              "episode_group_order_sha256": hashlib.sha256(
                  json.dumps(step_groups, separators=(",", ":")).encode()).hexdigest()}
    return report, seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.root / "data/qwen3_group4_exact512.parquet") != DATA_SHA:
        raise ValueError("exact512 train dataset changed")
    reports = []
    seed_sets = []
    for seed in (11, 22, 33):
        report, ids = one_seed(args.root, seed)
        reports.append(report)
        seed_sets.append(ids)
    if not (seed_sets[0] == seed_sets[1] == seed_sets[2]):
        raise ValueError("three control seeds used different episode IDs")
    if len({r["episode_group_order_sha256"] for r in reports}) != 1:
        raise ValueError("three control seeds used different group orders")
    result = {
        "schema": "oracle_exact512_control_group_signal_v1",
        "source_dataset_sha256": DATA_SHA,
        "seeds": reports,
        "mean_all_failure_group_fraction": statistics.mean(
            r["all_failure_group_fraction"] for r in reports),
        "mean_nonzero_terminal_variance_fraction": statistics.mean(
            r["nonzero_terminal_variance_fraction"] for r in reports),
        "mean_all_failure_same_mode_gap1m_group_fraction": statistics.mean(
            r["all_failure_same_mode_gap1m_group_fraction"] for r in reports),
        "interpretation": "R2R-train outcome-only group signal availability; all-failure groups have zero terminal reward contrast. Three seeds reuse the same 512 episode IDs. No candidate reward or val-unseen navigation result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mean_all_failure_group_fraction":
                      result["mean_all_failure_group_fraction"],
                      "mean_nonzero_terminal_variance_fraction":
                      result["mean_nonzero_terminal_variance_fraction"],
                      "by_seed": [(r["seed"], r["counts"]["all_failure_groups"])
                                  for r in reports]}))


if __name__ == "__main__":
    main()
