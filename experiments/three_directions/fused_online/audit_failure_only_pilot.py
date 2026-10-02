"""Check group-four failure-only reward wiring against matched controls."""

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
    parser.add_argument("--steps", type=int, choices=(2, 64), required=True)
    args = parser.parse_args()
    experiment = f"failure_only_group4_{args.steps}step_seed11"
    run = args.root / "runlogs" / experiment
    checkpoint = args.root / "verl_checkpoints" / experiment
    control = args.source_root / "verl_checkpoints" / \
        f"three_directions_group4_{args.steps}step_seed11"
    if not (run / "completed").is_file() or not \
            (checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file():
        raise ValueError("failure-only run incomplete")
    if digest(args.root / "data/branch_pilot_train.parquet") != \
            "a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3":
        raise ValueError("training dataset hash mismatch")
    candidate = [json.loads(row) for row in
                 (checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(row) for row in
                (control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != args.steps:
        raise ValueError("training step count mismatch")
    before = json.loads((run / "reward_health_before.json").read_text())
    after = json.loads((run / "reward_health_after.json").read_text())
    if before["reward_variant"] != "failure_only_temporal_v1" or \
            after["reward_variant"] != "failure_only_temporal_v1":
        raise ValueError("reward variant changed")
    if before["encoder_sha256"] != after["encoder_sha256"] or \
            before["calibration_sha256"] != after["calibration_sha256"]:
        raise ValueError("reward checkpoint or calibration changed")
    gradients = [float(value) for value in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or \
            not all(math.isfinite(value) for value in gradients[:args.steps]) or \
            not any(value > 1e-6 for value in gradients[:args.steps]):
        raise ValueError("actor gradient gate failed")
    matched = diverse = success = failure = all_failure_groups = 0
    scores = []
    seen_episodes = set()
    for step, (row, control_row) in enumerate(zip(candidate, baseline), 1):
        if row["step"] != step or control_row["step"] != step:
            raise ValueError("step order mismatch")
        by_episode = collections.defaultdict(list)
        for item in row["info"]:
            by_episode[str(item["episode_id"])].append(item)
        control_ids = collections.Counter(str(item["episode_id"])
                                          for item in control_row["info"])
        if set(by_episode) != set(control_ids) or \
                set(map(len, by_episode.values())) != {4} or \
                set(control_ids.values()) != {4} or \
                seen_episodes & set(by_episode):
            raise ValueError("episode or group-size mismatch")
        seen_episodes.update(by_episode)
        matched += len(by_episode)
        for items in by_episode.values():
            all_failure_groups += not any(item["task_success"] for item in items)
            diverse += len({tuple(turn["response"] for turn in item["gen_traj"])
                            for item in items}) > 1
            for item in items:
                components = item["reward_components"]
                bonus = float(components["fused_bonus"])
                if float(components["semantic_reward"]) != 0 or \
                        not math.isfinite(float(item["total_reward"])):
                    raise ValueError("unexpected reward component")
                if item["task_success"]:
                    success += 1
                    if bonus != 0 or item["fused_reward"]["status"] != "disabled" or \
                            float(components["success_floor"]) != 2:
                        raise ValueError("success rollout received failure-only bonus")
                else:
                    failure += 1
                    score = item["fused_reward"]
                    if score["status"] != "ok" or not 0 < bonus < 1 or \
                            abs(float(score["bonus"]) - bonus) > 1e-6 or \
                            float(components["success_floor"]) != 0 or \
                            not math.isfinite(float(score["raw"])):
                        raise ValueError("failure bonus invalid")
                    scores.append(bonus)
    if after["requests"] - before["requests"] != failure or \
            matched != 4 * args.steps or diverse == 0:
        raise ValueError("reward request or grouping coverage failed")
    result = {"schema": "failure_only_group4_training_wiring_audit_v1",
              "interpretation": "Training wiring only. No held-out navigation claim.",
              "steps": args.steps, "seed": 11, "group_size": 4,
              "matched_train_episode_sets": matched,
              "unique_train_episodes": len(seen_episodes),
              "rollouts": success + failure,
              "successful_rollouts": success,
              "unsuccessful_rollouts": failure,
              "reward_requests": after["requests"] - before["requests"],
              "all_failure_groups": all_failure_groups,
              "diverse_groups": diverse,
              "failure_bonus_range": [min(scores), max(scores)] if scores else None,
              "actor_grad_norms": gradients[:args.steps],
              "encoder_sha256": after["encoder_sha256"],
              "calibration_sha256": after["calibration_sha256"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
