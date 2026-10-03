"""Audit all 128 matched confidence/control steps before full evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from audit_confident_train import digest, expected_votes, HELPER_SHA, AGENT_SHA, GAP


DATA_SHA = "d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f"
MANIFEST_SHA = "1efa550bdd21ff947ff1dfd7e461a57fb056ac034bd8f7e801beb7064da56b0a"
EXPERT_SHA = "6e72f6b2a8c1b73216a1f01c70cb85fc733e592035799e3a4bbf12774edca1d4"


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
    manifest_path = (args.source_root /
                     "runlogs/ordinal_progress/qwen3_route_match/online_exact512_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != MANIFEST_SHA or \
            manifest["schema"] != "qwen3_group4_exact_start_scale_dataset_v1" or \
            manifest["selected_rows"] != 512 or \
            manifest["output_parquet_sha256"] != DATA_SHA or \
            digest(args.source_root / "data/qwen3_group4_exact512.parquet") != DATA_SHA or \
            digest(args.candidate_root / "data/qwen3_group4_exact512.parquet") != DATA_SHA or \
            digest(args.candidate_root /
                   "verl/workers/agent/qwen_group_reward.py") != HELPER_SHA or \
            digest(args.candidate_root /
                   "verl/workers/agent/parallel_env_vlnce.py") != AGENT_SHA:
        raise ValueError("scale dataset or reward-source provenance mismatch")
    names = {"control": f"qwen_confident_scale_control_128step_seed{args.seed}",
             "candidate": f"qwen_confident_scale_128step_seed{args.seed}"}
    roots = {"control": args.source_root, "candidate": args.candidate_root}
    runs, rollouts = {}, {}
    for arm in ("control", "candidate"):
        runs[arm] = roots[arm] / "runlogs" / names[arm]
        checkpoint = roots[arm] / "verl_checkpoints" / names[arm]
        if not (runs[arm] / "completed").is_file() or not (
                checkpoint / "global_step_128/actor/huggingface/config.json").is_file():
            raise ValueError(f"incomplete {arm} scale training")
        rollouts[arm] = [json.loads(line) for line in (
            checkpoint / "rollout.jsonl").read_text().splitlines()]
        if len(rollouts[arm]) != 128:
            raise ValueError(f"wrong {arm} step count")
    before = json.loads((runs["candidate"] / "reward_health_before.json").read_text())
    after = json.loads((runs["candidate"] / "reward_health_after.json").read_text())
    if before["variant"] != "qwen3_exact_start_group_rank_v1" or \
            after["variant"] != "qwen3_exact_start_group_rank_v1" or \
            before["manifest_sha256"] != MANIFEST_SHA or \
            after["manifest_sha256"] != MANIFEST_SHA or \
            before["expert_analysis_sha256"] != EXPERT_SHA or \
            after["expert_analysis_sha256"] != EXPERT_SHA:
        raise ValueError("scale reward service provenance mismatch")
    counts = Counter()
    seen = set()
    for step in range(1, 129):
        expected_ids = {str(row["episode_id"]) for row in
                        manifest["rows"][(step - 1) * 4:step * 4]}
        paired = {}
        for arm in ("control", "candidate"):
            row = rollouts[arm][step - 1]
            if row["step"] != step or len(row["info"]) != 16:
                raise ValueError("scale step or rollout-count mismatch")
            groups = defaultdict(list)
            for item in row["info"]:
                groups[str(item["episode_id"])].append(item)
                components = item["reward_components"]
                if not math.isfinite(float(item["total_reward"])) or \
                        float(components["ndtw_reward"]) != 0 or \
                        float(components["semantic_reward"]) != 0:
                    raise ValueError("nonfinite or unintended reward component")
                if arm == "control":
                    base_reward = sum(float(components[key]) for key in (
                        "success_reward", "success_floor", "ndtw_reward",
                        "semantic_reward"))
                    if float(components.get("fused_bonus", 0)) != 0 or \
                            float(components.get("qwen_group_ordinal", 0)) != 0 or \
                            abs(float(item["total_reward"]) - base_reward) > 1e-3:
                        raise ValueError("control received semantic bonus")
            if set(groups) != expected_ids or set(map(len, groups.values())) != {4}:
                raise ValueError(f"scale train row/group mismatch step {step} {arm}")
            paired[arm] = groups
        if seen.intersection(expected_ids):
            raise ValueError("scale training episode reused")
        seen.update(expected_ids)
        counts["groups"] += 4
        for items in paired["candidate"].values():
            counts["diverse_groups"] += len({tuple(
                turn["response"] for turn in item["gen_traj"]) for item in items}) > 1
            success = any(item["task_success"] for item in items)
            counts["all_failure_groups"] += not success
            for item in items:
                components = item["reward_components"]
                teacher = item["fused_reward"]
                ordinal = float(components["qwen_group_ordinal"])
                if not math.isfinite(ordinal) or abs(ordinal) > .5 + 1e-6 or \
                        float(components["fused_bonus"]) != 0 or \
                        float(teacher["removed_bonus"]) != 0 or \
                        abs(ordinal - float(teacher["applied_ordinal"])) > 1e-6:
                    raise ValueError("invalid confident scale reward wiring")
                if item["task_success"]:
                    counts["successes"] += 1
                    if teacher["status"] != "disabled" or ordinal != 0:
                        raise ValueError("success received teacher rank")
                else:
                    counts["failures_scored"] += 1
                    if teacher["status"] != "ok" or teacher["scored_views"] != 6:
                        raise ValueError("failure lacks frozen teacher score")
                total = sum(float(components[key]) for key in (
                    "success_reward", "success_floor", "ndtw_reward",
                    "semantic_reward", "qwen_group_ordinal"))
                if abs(float(item["total_reward"]) - total) > 1e-3:
                    raise ValueError("candidate total reward mismatch")
                counts["nonzero_ordinal_rollouts"] += ordinal != 0
            if success:
                if any(float(item["reward_components"]["qwen_group_ordinal"])
                       for item in items):
                    raise ValueError("mixed group received confidence reward")
                continue
            expected_votes_group, confident = expected_votes(items)
            counts["confident_pairs"] += confident
            counts["ordinal_active_groups"] += any(expected_votes_group)
            for item, value in zip(items, expected_votes_group):
                if abs(float(item["reward_components"]["qwen_group_ordinal"]) - value) > 1e-6:
                    raise ValueError("confidence reward disagrees with audit formula")
    if len(seen) != 512 or counts["groups"] != 512 or \
            counts["successes"] + counts["failures_scored"] != 2048 or \
            after["requests"] - before["requests"] != counts["failures_scored"] or \
            counts["diverse_groups"] == 0 or counts["ordinal_active_groups"] == 0:
        raise ValueError("scale outcome or teacher request coverage mismatch")
    report = {"schema": "qwen_confident_scale_train_audit_v1",
              "seed": args.seed, "steps": 128, "group_size": 4,
              "min_score_gap": GAP, "dataset_sha256": DATA_SHA,
              "unique_train_episodes": len(seen), "counts": dict(counts),
              "reward_requests": after["requests"] - before["requests"],
              "nonzero_gradient_steps": {
                  arm: gradients(runs[arm]) for arm in ("control", "candidate")},
              "interpretation": "Matched train-row and reward-wiring audit; no held-out navigation claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
