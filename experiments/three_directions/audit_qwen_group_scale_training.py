"""Audit paired 128-step group-four scale runs before full evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f"
MODES = ("stopped but goal not reached.", "number of turns exceeded.")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_ranks(items):
    ordered = sorted(float(item["fused_reward"]["raw"]) for item in items)
    center = (len(items) - 1) / 2
    return [(sum(index for index, value in enumerate(ordered)
                 if value == float(item["fused_reward"]["raw"])) /
             ordered.count(float(item["fused_reward"]["raw"])) - center) /
            (len(items) - 1) for item in items]


def gradients(run: Path):
    values = [float(item) for item in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(values) < 128 or not all(math.isfinite(x) and x >= 0 for x in values[:128]) or \
            not any(x > 1e-6 for x in values[:128]):
        raise ValueError("optimizer gradient audit failed")
    return sum(x > 1e-6 for x in values[:128])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_root /
                           "runlogs/ordinal_progress/qwen3_route_match/online_exact512_manifest.json").read_text())
    if manifest["schema"] != "qwen3_group4_exact_start_scale_dataset_v1" or \
            manifest["selected_rows"] != 512 or \
            manifest["output_parquet_sha256"] != DATA_SHA or \
            digest(args.source_root / "data/qwen3_group4_exact512.parquet") != DATA_SHA or \
            digest(args.candidate_root / "data/qwen3_group4_exact512.parquet") != DATA_SHA:
        raise ValueError("scale dataset provenance mismatch")
    names = {"control": f"qwen_exact_scale_control_128step_seed{args.seed}",
             "candidate": f"qwen_group_rank_scale_128step_seed{args.seed}"}
    roots = {"control": args.source_root, "candidate": args.candidate_root}
    runs, checkpoints, rollouts = {}, {}, {}
    for arm in ("control", "candidate"):
        runs[arm] = roots[arm] / "runlogs" / names[arm]
        checkpoints[arm] = roots[arm] / "verl_checkpoints" / names[arm]
        if not (runs[arm] / "completed").is_file() or not (
                checkpoints[arm] /
                "global_step_128/actor/huggingface/config.json").is_file():
            raise ValueError(f"incomplete {arm} scale training")
        rollouts[arm] = [json.loads(line) for line in (
            checkpoints[arm] / "rollout.jsonl").read_text().splitlines()]
        if len(rollouts[arm]) != 128:
            raise ValueError(f"wrong {arm} step coverage")
    before = json.loads((runs["candidate"] /
                         "reward_health_before.json").read_text())
    after = json.loads((runs["candidate"] /
                        "reward_health_after.json").read_text())
    if before["variant"] != "qwen3_exact_start_group_rank_v1" or \
            after["variant"] != "qwen3_exact_start_group_rank_v1" or \
            before["manifest_sha256"] != after["manifest_sha256"] or \
            before["expert_analysis_sha256"] != after["expert_analysis_sha256"]:
        raise ValueError("reward service changed during scale training")
    counts = Counter()
    seen = set()
    for step in range(1, 129):
        grouped = {}
        expected = {row["episode_id"] for row in
                    manifest["rows"][(step - 1) * 4:step * 4]}
        for arm in ("control", "candidate"):
            row = rollouts[arm][step - 1]
            if row["step"] != step or len(row["info"]) != 16:
                raise ValueError("step/rollout count mismatch")
            groups = defaultdict(list)
            for item in row["info"]:
                groups[str(item["episode_id"])].append(item)
                components = item["reward_components"]
                if not math.isfinite(float(item["total_reward"])) or \
                        float(components["ndtw_reward"]) != 0 or \
                        float(components["semantic_reward"]) != 0:
                    raise ValueError("nonfinite or unintended reward component")
                if arm == "control":
                    expected_control = sum(float(components[key]) for key in (
                        "success_reward", "success_floor", "ndtw_reward",
                        "semantic_reward"))
                    if float(components.get("fused_bonus", 0)) != 0 or \
                            float(components.get("qwen_group_ordinal", 0)) != 0 or \
                            abs(float(item["total_reward"]) - expected_control) > 1e-3:
                        raise ValueError("control contains semantic reward")
            if set(groups) != expected or set(map(len, groups.values())) != {4}:
                raise ValueError(f"training row/group mismatch step {step} {arm}")
            grouped[arm] = groups
        if seen.intersection(expected):
            raise ValueError("scale train episode reused")
        seen.update(expected)
        counts["groups"] += 4
        for eid, items in grouped["candidate"].items():
            counts["diverse_groups"] += len({tuple(turn["response"] for turn in item["gen_traj"])
                                              for item in items}) > 1
            success = any(item["task_success"] for item in items)
            counts["all_failure_groups"] += not success
            for item in items:
                components = item["reward_components"]
                teacher = item["fused_reward"]
                ordinal = float(components["qwen_group_ordinal"])
                if not math.isfinite(ordinal) or abs(ordinal) > .5 or \
                        components["fused_bonus"] != 0 or \
                        float(teacher["removed_bonus"]) != 0 or \
                        abs(ordinal - float(teacher["applied_ordinal"])) > 1e-6:
                    raise ValueError("invalid group ordinal wiring")
                if item["task_success"]:
                    counts["successes"] += 1
                    if teacher["status"] != "disabled" or ordinal != 0:
                        raise ValueError("success received teacher reward")
                else:
                    counts["failures_scored"] += 1
                    if teacher["status"] != "ok" or teacher["scored_views"] != 6:
                        raise ValueError("failure missing teacher score")
                expected_reward = sum(float(components[key]) for key in (
                    "success_reward", "success_floor", "ndtw_reward",
                    "semantic_reward", "qwen_group_ordinal"))
                if abs(float(item["total_reward"]) - expected_reward) > 1e-3:
                    raise ValueError("reward total mismatch")
                counts["nonzero_ordinal_rollouts"] += ordinal != 0
            if success:
                if any(float(item["reward_components"]["qwen_group_ordinal"])
                       for item in items):
                    raise ValueError("mixed group got teacher ordinal")
                continue
            for mode in MODES:
                members = [item for item in items if item["end_reason"] == mode]
                if len(members) < 2:
                    if any(float(item["reward_components"]["qwen_group_ordinal"])
                           for item in members):
                        raise ValueError("singleton failure received ordinal")
                    continue
                actual = [float(item["reward_components"]["qwen_group_ordinal"])
                          for item in members]
                reference = expected_ranks(members)
                if any(abs(a - b) > 1e-6 for a, b in zip(actual, reference)) or \
                        abs(sum(actual)) > 1e-6:
                    raise ValueError("same-mode rank or zero-sum mismatch")
                counts["same_mode_pairs"] += len(members) * (len(members) - 1) // 2
    if len(seen) != 512 or counts["groups"] != 512 or \
            counts["successes"] + counts["failures_scored"] != 2048 or \
            after["requests"] - before["requests"] != counts["failures_scored"] or \
            counts["diverse_groups"] == 0 or counts["nonzero_ordinal_rollouts"] == 0:
        raise ValueError("scale outcome/teacher coverage mismatch")
    report = {"schema": "qwen_group_rank_scale_train_audit_v1",
              "seed": args.seed, "steps": 128, "rollout_n": 4,
              "dataset_sha256": DATA_SHA,
              "unique_train_episodes": len(seen),
              "counts": dict(counts),
              "reward_requests": after["requests"] - before["requests"],
              "nonzero_gradient_steps": {
                  arm: gradients(runs[arm]) for arm in ("control", "candidate")},
              "interpretation": "Matched train-row and reward wiring audit; no held-out navigation claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
