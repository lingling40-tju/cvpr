"""Audit stop-conditioned reward requests and matched group-four training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3"
ENV_SHA = "e3644b522cf630f489da049eb6766eb02835cb6fda898a494e951d60e4048051"
STOP_REASON = "stopped but goal not reached."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 64, 128), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), default=11)
    args = parser.parse_args()
    if args.steps < 128 and args.seed != 11:
        raise ValueError("two/64-step matched controls are seed 11 only")
    name = f"stopaware_group4_{args.steps}step_seed{args.seed}"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control_name = f"three_directions_group4_{args.steps}step"
    if args.steps < 128 or args.seed != 11:
        control_name += f"_seed{args.seed}"
    control = args.source_root / "verl_checkpoints" / control_name
    if not (run / "completed").is_file() or not \
            (checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file() or \
            digest(args.root / "data/branch_pilot_train.parquet") != DATA_SHA or \
            digest(args.root / "vlnce_server/semantic_reward/env.py") != ENV_SHA:
        raise ValueError("incomplete run or provenance mismatch")
    candidate = [json.loads(line) for line in (checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (control / "rollout.jsonl").read_text().splitlines()]
    if len(candidate) != args.steps or len(baseline) != args.steps:
        raise ValueError("training step count mismatch")
    before = json.loads((run / "reward_health_before.json").read_text())
    after = json.loads((run / "reward_health_after.json").read_text())
    if before["reward_variant"] != "failure_only_temporal_v1" or \
            after["reward_variant"] != "failure_only_temporal_v1" or \
            before["encoder_sha256"] != after["encoder_sha256"] or \
            before["calibration_sha256"] != after["calibration_sha256"]:
        raise ValueError("scorer changed during training")
    gradients = [float(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < args.steps or not all(math.isfinite(x) for x in gradients[:args.steps]) or \
            not any(x > 1e-6 for x in gradients[:args.steps]):
        raise ValueError("actor gradient audit failed")
    groups = diverse = successes = failures = requests = positive_bonus = 0
    censored = Counter()
    seen_ids = set()
    for step, (row, base) in enumerate(zip(candidate, baseline), 1):
        if row["step"] != step or base["step"] != step:
            raise ValueError("training step order mismatch")
        by_episode = defaultdict(list)
        for item in row["info"]:
            by_episode[str(item["episode_id"])].append(item)
        base_ids = Counter(str(item["episode_id"]) for item in base["info"])
        if set(by_episode) != set(base_ids) or set(map(len, by_episode.values())) != {4} or \
                set(base_ids.values()) != {4} or seen_ids & set(by_episode):
            raise ValueError("matched episode/group-four coverage mismatch")
        seen_ids.update(by_episode)
        groups += len(by_episode)
        for items in by_episode.values():
            diverse += len({tuple(turn["response"] for turn in item["gen_traj"])
                            for item in items}) > 1
            for item in items:
                comp = item["reward_components"]
                bonus = float(comp["fused_bonus"])
                score = item["fused_reward"]
                if float(comp["semantic_reward"]) != 0 or \
                        not math.isfinite(float(item["total_reward"])) or \
                        not 0 <= bonus <= 1:
                    raise ValueError("invalid reward component")
                if item["task_success"]:
                    successes += 1
                    if score["status"] != "disabled" or bonus != 0 or \
                            float(comp["success_floor"]) != 2:
                        raise ValueError("success received stop-aware bonus")
                elif item["end_reason"] == STOP_REASON:
                    failures += 1
                    requests += 1
                    raw = float(score["bonus"])
                    expected = max(0.0, 2 * raw - 1)
                    if score["status"] != "ok" or not 0 <= raw <= 1 or \
                            abs(bonus - expected) > 1e-6 or \
                            abs(float(score["applied_bonus"]) - bonus) > 1e-6 or \
                            float(comp["success_floor"]) != 0:
                        raise ValueError("stop-conditioned bonus mismatch")
                    positive_bonus += bonus > 0
                else:
                    failures += 1
                    censored[str(item["end_reason"])] += 1
                    if score["status"] != "censored" or bonus != 0 or \
                            score["reason"] != item["end_reason"]:
                        raise ValueError("timeout/format failure received bonus")
    if after["requests"] - before["requests"] != requests or \
            groups != 4 * args.steps or successes + failures != 16 * args.steps or \
            diverse == 0:
        raise ValueError("service or group coverage mismatch")
    result = {"schema": "stopaware_group4_training_wiring_audit_v1",
              "interpretation": "Train-scene reward wiring only. No held-out navigation claim.",
              "steps": args.steps, "seed": args.seed, "group_size": 4,
              "matched_unique_episode_groups": groups,
              "rollouts": successes + failures,
              "successful_rollouts_without_bonus": successes,
              "unsuccessful_voluntary_stops_scored": requests,
              "unsuccessful_rollouts": failures,
              "censored_failure_reasons": dict(censored),
              "positive_centered_bonus_rollouts": positive_bonus,
              "reward_requests": after["requests"] - before["requests"],
              "diverse_groups": diverse,
              "nonzero_actor_gradient_steps": sum(x > 1e-6 for x in gradients[:args.steps]),
              "env_sha256": ENV_SHA,
              "encoder_sha256": after["encoder_sha256"],
              "calibration_sha256": after["calibration_sha256"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
