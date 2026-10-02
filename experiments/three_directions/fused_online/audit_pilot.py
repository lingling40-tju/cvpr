"""Audit fused group-four reward wiring against the matched control."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64, 128), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), default=11)
    args = parser.parse_args()
    experiment = f"fused_reward_group4_{args.steps}step_seed{args.seed}"
    run = args.root / "runlogs" / experiment
    checkpoint = args.root / "verl_checkpoints" / experiment
    control_name = f"three_directions_group4_{args.steps}step"
    if args.steps < 128 or args.seed != 11:
        control_name += f"_seed{args.seed}"
    control = args.source_root / "verl_checkpoints" / control_name
    if not (run / "completed").is_file() or not \
            (checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file():
        raise ValueError("fused run incomplete")
    if digest(args.root / "data/branch_pilot_train.parquet") != \
            "a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3":
        raise ValueError("training data hash mismatch")
    candidate = [json.loads(row) for row in
                 (checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(row) for row in
                (control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != args.steps:
        raise ValueError("training step count mismatch")
    before = json.loads((run / "reward_health_before.json").read_text())
    after = json.loads((run / "reward_health_after.json").read_text())
    if before["development_report_sha256"] != \
            after["development_report_sha256"]:
        raise ValueError("reward service changed during training")
    bonus = []
    scores = []
    gradients = [float(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or not all(math.isfinite(x) for x in gradients[:args.steps]) or \
            not any(x > 1e-6 for x in gradients[:args.steps]):
        raise ValueError("actor gradient gate failed")
    matched = diverse = 0
    episode_ids = set()
    for step, (row, control_row) in enumerate(zip(candidate, baseline), 1):
        if row["step"] != step or control_row["step"] != step:
            raise ValueError("step index mismatch")
        by_episode = collections.defaultdict(list)
        control_by_episode = collections.Counter(str(item["episode_id"])
                                                 for item in control_row["info"])
        for item in row["info"]:
            eid = str(item["episode_id"])
            by_episode[eid].append(item)
            components = item["reward_components"]
            fused = item.get("fused_reward", {})
            value = float(components["fused_bonus"])
            if fused.get("status") != "ok" or not 0 < value < 1 or \
                    abs(float(fused["bonus"]) - value) > 1e-6 or \
                    not math.isfinite(float(item["total_reward"])) or \
                    float(components["semantic_reward"]) != 0:
                raise ValueError(f"invalid fused reward on step {step}, episode {eid}")
            bonus.append(value)
            scores.append(float(fused["raw"]))
        if set(by_episode) != set(control_by_episode) or \
                set(map(len, by_episode.values())) != {4} or \
                set(control_by_episode.values()) != {4}:
            raise ValueError(f"group-four episode mismatch at step {step}")
        if episode_ids & set(by_episode):
            raise ValueError("reused training episode")
        episode_ids.update(by_episode)
        matched += len(by_episode)
        diverse += sum(len({tuple(turn["response"] for turn in item["gen_traj"])
                            for item in items}) > 1
                       for items in by_episode.values())
    if after["requests"] - before["requests"] != len(bonus) or diverse == 0:
        raise ValueError("reward request coverage or rollout diversity failed")
    result = {"schema": "fused_group4_training_wiring_audit_v1",
              "interpretation": "Training wiring only; not held-out navigation evidence.",
              "steps": args.steps, "seed": args.seed, "group_size": 4,
              "matched_train_episode_sets": matched,
              "unique_train_episodes": len(episode_ids),
              "rollouts": len(bonus), "reward_requests": after["requests"] - before["requests"],
              "diverse_groups": diverse,
              "fused_bonus_range": [min(bonus), max(bonus)],
              "fused_raw_range": [min(scores), max(scores)],
              "actor_grad_norms": gradients[:args.steps],
              "development_report_sha256": before["development_report_sha256"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
