"""Reconstruct trajectory credit from a completed candidate's recorded rewards.

CPU-only descriptive analysis. Alternative rules use the same recorded
trajectories; they are not policy-training or navigation ablations.
"""

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re


SOURCE = {
    "verl/trainer/ppo/positive_trajectory_advantage.py": "d7516263c565a276797b8f6b7d9d5d3976ae81f2d5ea47331870e8404f9ead69",
    "verl/trainer/ppo/ray_trainer.py": "50356c80a1fda653e10f4332d128b611d29a4268598023d37c66513c26f31cdb",
    "verl/workers/agent/parallel_env_vlnce.py": "903c3788902a613c68d8d4c0f681eacbe3d0d4eb421c7ea4d2a45f4cde648391",
    "verl/workers/reward_manager/naive.py": "33588f02f58dd4bcec31b10d4451e1540c373b2dc49405e3c684ddc19afa248c",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    p.add_argument("--tensorboard", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility must be disabled")
    require(not a.compact.exists() and not a.output.exists(), "Output already exists")
    for name, digest in SOURCE.items():
        require(sha(a.root / name) == digest, "Inspected source changed: " + name)
    import torch
    torch.set_num_threads(1)
    module_path = a.root / "verl/trainer/ppo/positive_trajectory_advantage.py"
    spec = importlib.util.spec_from_file_location("recorded_positive_weight_rule", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constants = {key: getattr(module, key) for key in
                 ("GROUP_SIZE", "SUCCESS_PRIORITY", "FAILURE_SCORE_FLOOR", "GAP_SCALE", "MAX_ADVANTAGE")}
    require(constants == {"GROUP_SIZE": 4, "SUCCESS_PRIORITY": 15.0,
                          "FAILURE_SCORE_FLOOR": 2.5, "GAP_SCALE": 5.0, "MAX_ADVANTAGE": 1.5},
            "Frozen rule differs")
    label = f"positive_trajectory_candidate_128step_seed{a.seed}"
    run = a.root / "runlogs" / label
    require((run / "completed").is_file() and not (run / "failed").exists(), "Candidate incomplete")
    audit_path = a.root / f"runlogs/positive_scale/candidate_seed{a.seed}_train_audit.json"
    audit = json.loads(audit_path.read_text())
    require(audit["arm"] == "candidate" and audit["observed_steps"] == audit["expected_steps"] == 128,
            "Training audit differs")
    log_path = run / "train.log"
    log = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace"))
    require(sha(log_path) == audit["train_log_sha256"], "Audited log changed")
    require("'use_kl_in_reward': False" in log and "'reward_manager': 'naive'" in log,
            "Reward configuration differs")
    require(re.search(r"'custom_reward_function':\s*\{[^}]*'path': None", log) is not None,
            "Unexpected custom reward function")
    tb = json.loads(a.tensorboard.read_text())
    require(tb["run"] == label and tb["arm"] == "candidate" and tb["seed"] == a.seed and
            tb["train_log_sha256"] == sha(log_path) and tb["training_audit_sha256"] == sha(audit_path),
            "TensorBoard provenance differs")
    scalars = {row["step"]: row for row in tb["steps"]}
    require(sorted(scalars) == list(range(1, 129)), "Scalar coverage differs")
    fit_path = a.root / "prepared_data/fit512_manifest.json"
    fit = set(json.loads(fit_path.read_text())["episode_ids"])
    require(len(fit) == 512 and sha(fit_path) == "98dd7ec7bcf707dcfe62162761b26a3ff6f1170101bdbccf1c19e0ec0d18bc20",
            "Fit identity differs")
    raw_path = a.root / f"verl_checkpoints/{label}/rollout.jsonl"
    original_raw_sha = sha(raw_path)
    rows, counts, changes, differences = [], Counter(), Counter(), Counter()
    success_group_counts = Counter()
    max_scalar_error = Counter()
    last_step = 0

    def weights(scores, success, groups, override=None):
        try:
            if override:
                for key, value in override.items(): setattr(module, key, value)
            w, ret = module.positive_trajectory_advantage(scores, success, torch.ones((32, 1)), groups)
            require(torch.equal(w, ret), "Return copy differs")
            return w[:, 0].tolist()
        finally:
            for key, value in constants.items(): setattr(module, key, value)

    for line in raw_path.open():
        record = json.loads(line)
        step, info = record["step"], record["info"]
        require(step == last_step + 1 and len(info) == 32, "Raw step coverage differs")
        last_step = step
        ids = [str(i["episode_id"]) for i in info]
        groups = defaultdict(list)
        for i, episode in enumerate(ids): groups[episode].append(i)
        require(len(groups) == 8 and all(len(i) == 4 for i in groups.values()) and set(groups) <= fit,
                "Raw episode groups differ")
        score_values, success_values = [], []
        for item in info:
            score = float(item["total_reward"])
            rewards = [float(t["reward"]) for t in item["gen_traj"]]
            require(item["done"] is True and type(item["task_success"]) is bool and
                    math.isfinite(score) and 0 <= score <= 20 and
                    all(math.isfinite(x) and x >= 0 for x in rewards), "Invalid terminal record")
            require(math.isclose(sum(rewards), score, rel_tol=0, abs_tol=1e-6) and
                    sum(x > 0 for x in rewards) <= 1, "Recorded reward path differs")
            score_values.append(score); success_values.append(int(item["task_success"]))
        scores, success = torch.tensor(score_values, dtype=torch.float32), torch.tensor(success_values)
        actual = weights(scores, success, ids)
        variants = {
            "no_success_priority": weights(scores, success, ids, {"SUCCESS_PRIORITY": 0.0}),
            "no_failure_floor": weights(scores, success, ids, {"FAILURE_SCORE_FLOOR": 0.0}),
            "no_cap": weights(scores, success, ids, {"MAX_ADVANTAGE": float("inf")}),
            "constant_success_proxy": [1.5 * x for x in success_values],
        }
        checks = {"critic/score/mean": float(scores.mean()), "critic/score/max": float(scores.max()),
                  "critic/advantages/min": 0.0, "critic/advantages/max": max(actual),
                  "agent_reason/[successfully reached the goal.]": sum(success_values) / 32}
        for key, value in checks.items():
            error = abs(value - scalars[step][key])
            require(error <= 1e-5, f"Reconstructed {key} differs from stored scalar at step {step}")
            max_scalar_error[key] = max(max_scalar_error[key], error)
        for episode, indices in sorted(groups.items()):
            n_success = sum(success_values[i] for i in indices)
            success_group_counts[str(n_success)] += 1
            counts["groups"] += 1
            counts["all_failure_groups_with_positive_credit"] += n_success == 0 and any(actual[i] > 0 for i in indices)
            rows.append({"step": step, "episode_id": episode, "recorded_terminal_scores": [score_values[i] for i in indices],
                         "score_float32": [float(scores[i]) for i in indices],
                         "task_success": [success_values[i] for i in indices],
                         "reconstructed_weights": [actual[i] for i in indices],
                         "same_record_alternative_weights": {key: [value[i] for i in indices] for key, value in variants.items()}})
        for i, w in enumerate(actual):
            s = success_values[i]
            counts["trajectories"] += 1; counts["successful_trajectories"] += s
            counts["failed_positive_score"] += not s and score_values[i] > 0
            counts["failed_floor_eligible"] += not s and float(scores[i]) >= 2.5
            counts["positive_credit_trajectories"] += w > 0
            counts["positive_credit_successes"] += bool(s and w > 0)
            counts["positive_credit_failures"] += bool(not s and w > 0)
            counts["successful_zero_credit"] += bool(s and w == 0)
            counts["capped_positive_credit"] += w == 1.5
            counts["uncapped_positive_credit"] += 0 < w < 1.5
            for key, value in variants.items():
                delta = abs(w - value[i]); changes[key] += delta > 1e-5; differences[key] += delta
    require(last_step == 128 and counts["groups"] == 1024 and counts["trajectories"] == 4096,
            "Complete budget differs")
    require(sha(raw_path) == original_raw_sha, "Original rollout changed while reading")
    for name, digest in SOURCE.items(): require(sha(a.root / name) == digest, "Source changed while reading")
    a.compact.parent.mkdir(parents=True, exist_ok=True)
    with a.compact.open("x") as handle:
        handle.write("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    report = {"schema": "positive_recorded_credit_mechanism_v1", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "run": label, "configured_seed": a.seed, "steps": 128, "constants": constants,
              "counts": dict(counts), "groups_by_number_of_successes": dict(success_group_counts),
              "same_record_changed_weight_counts": dict(changes),
              "same_record_absolute_weight_change_sums": dict(differences),
              "maximum_error_vs_stored_scalars": dict(max_scalar_error),
              "training_audit_sha256": sha(audit_path), "train_log_sha256": sha(log_path),
              "tensorboard_export_sha256": sha(a.tensorboard), "fit_manifest_sha256": sha(fit_path),
              "rollout_sha256": original_raw_sha, "source_sha256": SOURCE,
              "default_reward_dispatch_sha256": sha(a.root / 'verl/utils/reward_score/__init__.py'),
              "auditor_sha256": sha(Path(__file__)), "compact_sha256": sha(a.compact),
              "scope": "Reconstructed per-trajectory credit on actual recorded training rewards, matching stored step score/advantage maxima and success frequencies. Not saved token advantages, gradients, policy ablations, semantic truth or navigation gains. Alternative rules keep trajectories fixed."}
    with a.output.open("x") as handle: handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run": label, "counts": dict(counts), "same_record_changed_weight_counts": dict(changes)}))


if __name__ == "__main__":
    main()
