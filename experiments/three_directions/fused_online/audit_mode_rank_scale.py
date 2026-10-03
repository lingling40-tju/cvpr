"""Audit matched group-four training and mode-stratified ordinal rewards."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from mode_stratified_reward import ELIGIBLE_MODES


DATA_SHA = "a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3"
AGENT_SHA = "5a35ed8f48337b8362a44e6ef31e5e4ed546c481372aba8c6ab6d72433c094fa"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_ranks(members):
    scores = [float(item["fused_reward"]["raw"]) for item in members]
    ordered = sorted(scores)
    center = (len(members) - 1) / 2
    return [(sum(index for index, value in enumerate(ordered) if value == score) /
             ordered.count(score) - center) / (len(members) - 1)
            for score in scores]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(128,), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"mode_rank_group4_{args.steps}step_seed{args.seed}"
    run = args.root / "runlogs" / name
    checkpoint = args.root / "verl_checkpoints" / name
    control_name = "three_directions_group4_128step"
    if args.seed != 11:
        control_name += f"_seed{args.seed}"
    control = args.source_root / "verl_checkpoints" / control_name
    if not (run / "completed").is_file() or not \
            (checkpoint / f"global_step_{args.steps}/actor/huggingface/config.json").is_file() or \
            digest(args.root / "data/branch_pilot_train.parquet") != DATA_SHA or \
            digest(args.root / "verl/workers/agent/parallel_env_vlnce.py") != AGENT_SHA:
        raise ValueError("incomplete run or provenance mismatch")
    candidate = [json.loads(line) for line in (checkpoint / "rollout.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (control / "rollout.jsonl").read_text().splitlines()]
    assert len(candidate) == len(baseline) == args.steps
    before = json.loads((run / "reward_health_before.json").read_text())
    after = json.loads((run / "reward_health_after.json").read_text())
    assert before["reward_variant"] == after["reward_variant"] == "failure_only_temporal_v1"
    assert before["encoder_sha256"] == after["encoder_sha256"]
    assert before["calibration_sha256"] == after["calibration_sha256"]
    gradients = [float(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    assert len(gradients) >= args.steps and all(
        math.isfinite(x) and x > 1e-6 for x in gradients[:args.steps])
    counts = Counter()
    seen = set()
    for step, (row, base) in enumerate(zip(candidate, baseline), 1):
        assert row["step"] == base["step"] == step
        groups = defaultdict(list)
        for item in row["info"]:
            groups[str(item["episode_id"])].append(item)
        base_ids = Counter(str(item["episode_id"]) for item in base["info"])
        assert set(groups) == set(base_ids) and not seen.intersection(groups)
        assert set(map(len, groups.values())) == {4} and set(base_ids.values()) == {4}
        seen.update(groups)
        counts["groups"] += len(groups)
        for items in groups.values():
            counts["diverse_groups"] += len({tuple(turn["response"] for turn in item["gen_traj"])
                                              for item in items}) > 1
            successful = any(item["task_success"] for item in items)
            counts["all_failure_groups"] += not successful
            for item in items:
                components = item["reward_components"]
                reward = item["fused_reward"]
                ordinal = float(components["mode_ordinal"])
                assert abs(ordinal - float(reward["applied_ordinal"])) < 1e-6
                assert components["fused_bonus"] == 0
                assert math.isfinite(ordinal) and abs(ordinal) <= .5
                if item["task_success"]:
                    counts["successes"] += 1
                    assert reward["status"] == "disabled" and \
                           reward["removed_bonus"] == ordinal == 0
                else:
                    counts["failures_scored"] += 1
                    assert reward["status"] == "ok" and \
                           abs(float(reward["removed_bonus"]) -
                               float(reward["bonus"])) < 1e-6
                expected = sum(float(components[key]) for key in (
                    "success_reward", "success_floor", "ndtw_reward",
                    "semantic_reward", "mode_ordinal"))
                assert abs(float(item["total_reward"]) - expected) < 1e-3
                counts["nonzero_ordinal_rollouts"] += ordinal != 0
            if successful:
                assert all(item["reward_components"]["mode_ordinal"] == 0
                           for item in items)
                continue
            active = False
            for mode in ELIGIBLE_MODES:
                members = [item for item in items if item["end_reason"] == mode]
                if len(members) < 2:
                    assert all(item["reward_components"]["mode_ordinal"] == 0
                               for item in members)
                    continue
                actual = [float(item["reward_components"]["mode_ordinal"])
                          for item in members]
                reference = expected_ranks(members)
                assert all(abs(a - b) < 1e-6 for a, b in zip(actual, reference))
                assert abs(sum(actual)) < 1e-6
                active |= any(actual)
                counts["same_mode_pairs"] += len(members) * (len(members) - 1) // 2
            counts["ordinal_active_groups"] += active
    assert counts["groups"] == 4 * args.steps
    assert counts["successes"] + counts["failures_scored"] == 16 * args.steps
    assert after["requests"] - before["requests"] == counts["failures_scored"]
    assert counts["diverse_groups"] > 0 and counts["ordinal_active_groups"] > 0
    report = {
        "schema": "mode_stratified_group4_training_audit_v1",
        "interpretation": "Matched train-scene reward wiring only; no val-unseen navigation claim.",
        "steps": args.steps,
        "seed": args.seed,
        "group_size": 4,
        "counts": dict(counts),
        "reward_requests": after["requests"] - before["requests"],
        "nonzero_actor_gradient_steps": args.steps,
        "agent_sha256": AGENT_SHA,
        "encoder_sha256": after["encoder_sha256"],
        "calibration_sha256": after["calibration_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
