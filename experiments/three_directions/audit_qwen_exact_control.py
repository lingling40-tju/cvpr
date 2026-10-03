"""Check exact-row n=4 outcome-only training before candidate launch."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.root / "runlogs/qwen3_exact_control_64step_seed11"
    checkpoint = args.root / "verl_checkpoints/qwen3_exact_control_64step_seed11"
    manifest = json.loads((args.root / "runlogs/ordinal_progress/qwen3_route_match/online_exact256_manifest.json").read_text())
    if not (run / "completed").is_file() or \
            not (checkpoint / "global_step_64/actor/huggingface/config.json").is_file() or \
            hashlib.sha256((args.root / "data/qwen3_group4_exact256.parquet").read_bytes()).hexdigest() != DATA_SHA or \
            manifest["output_parquet_sha256"] != DATA_SHA:
        raise ValueError("control completion/data provenance mismatch")
    rows = [json.loads(line) for line in (checkpoint / "rollout.jsonl").read_text().splitlines()]
    if len(rows) != 64 or len(manifest["rows"]) != 256:
        raise ValueError("control rollout coverage mismatch")
    seen = set()
    diverse = 0
    all_failure = eligible_all_failure = same_mode_pairs = 0
    for step, row in enumerate(rows, 1):
        if row["step"] != step or len(row["info"]) != 16:
            raise ValueError("control step/group size mismatch")
        groups = {}
        for info in row["info"]:
            eid = str(info["episode_id"])
            groups.setdefault(eid, []).append(info)
            components = info["reward_components"]
            if not math.isfinite(float(info["total_reward"])) or \
                    float(components["ndtw_reward"]) != 0 or \
                    float(components["semantic_reward"]) != 0:
                raise ValueError("invalid outcome-only reward")
        expected = {item["episode_id"] for item in
                    manifest["rows"][(step - 1) * 4:step * 4]}
        if set(groups) != expected or set(map(len, groups.values())) != {4} or \
                seen.intersection(groups):
            raise ValueError("training rows differ from frozen dataset")
        seen.update(groups)
        diverse += sum(len({tuple(turn["response"] for turn in item["gen_traj"])
                            for item in group}) > 1 for group in groups.values())
        for group in groups.values():
            if any(item["task_success"] for item in group):
                continue
            all_failure += 1
            modes = Counter(item["end_reason"] for item in group)
            pairs = sum(count * (count - 1) // 2 for mode, count in modes.items()
                        if mode in ("stopped but goal not reached.",
                                    "number of turns exceeded."))
            eligible_all_failure += pairs > 0
            same_mode_pairs += pairs
    gradients = [float(x) for x in re.findall(
        r"actor/grad_norm:([0-9.eE+-]+)", (run / "train.log").read_text())]
    if len(gradients) < 64 or not any(x > 1e-6 for x in gradients[:64]) or \
            not all(math.isfinite(x) and x >= 0 for x in gradients[:64]):
        raise ValueError("control optimizer audit failed")
    if seen != {row["episode_id"] for row in manifest["rows"]} or diverse == 0:
        raise ValueError("control uniqueness/diversity failed")
    report = {"schema": "qwen3_exact_n4_control_training_audit_v1",
              "steps": 64, "seed": 11, "group_size": 4,
              "unique_training_episodes": len(seen), "rollouts": 1024,
              "diverse_groups": diverse,
              "all_failure_groups": all_failure,
              "eligible_all_failure_groups": eligible_all_failure,
              "same_mode_failure_pairs": same_mode_pairs,
              "nonzero_gradient_steps": sum(x > 1e-6 for x in gradients[:64]),
              "dataset_sha256": DATA_SHA,
              "interpretation": "Matched train-row audit only; no held-out navigation result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
