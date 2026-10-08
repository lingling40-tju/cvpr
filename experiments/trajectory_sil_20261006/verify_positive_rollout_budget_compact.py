#!/usr/bin/env python3
"""Independently tally copied per-step training membership, without raw inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Local trajectory_sil_20261006 evidence directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    manifest_path = args.root / "fit512_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fit = set(manifest["episode_ids"])
    require(len(fit) == len(manifest["episode_ids"]) == 512, "Fit IDs differ")
    source_sha = digest(args.root / "audit_positive_rollout_budget.py")
    checked = []
    batches_by_arm_seed = {}
    paths = sorted((args.root / "scale128").glob("*_rollout_budget.json"))
    require(bool(paths), "No budget exports found")
    for path in paths:
        d = json.loads(path.read_text())
        arm, seed = d["arm"], d["seed"]
        require(arm in ("control", "candidate") and seed in (11, 22, 33), "Unknown arm or seed")
        label = "positive_trajectory_{}_128step_seed{}".format(arm, seed)
        require(d["schema"] == "positive_trajectory_recorded_rollout_budget_v1" and
                d["run"] == label and (arm, seed) not in batches_by_arm_seed, "Export identity differs")
        require(d["fit_manifest_sha256"] == digest(manifest_path) and
                d["fit_parquet_sha256"] == manifest["generated_parquet_sha256"] and
                d["auditor_sha256"] == source_sha, "Manifest or raw auditor identity differs")
        stem = "{}_seed{}".format(arm, seed)
        audit_path = path.parent / (stem + "_train_audit.json")
        audit = json.loads(audit_path.read_text())
        tb = json.loads((path.parent / (stem + "_tensorboard.json")).read_text())
        require(d["training_audit_sha256"] == digest(audit_path) == tb["training_audit_sha256"] and
                d["train_log_sha256"] == audit["train_log_sha256"] == tb["train_log_sha256"] and
                d["training_config_sha256"] == tb["config_sha256"] and
                d["completed"] == tb["completed_at_utc"] and
                tb["run"] == label and tb["arm"] == arm and tb["seed"] == seed,
                "Copied optimizer, scalar, and budget provenance disagree")
        rows = d["steps"]
        require([row["step"] for row in rows] == list(range(1, 129)), "Step coverage differs")
        epochs = [Counter(), Counter()]
        for step, row in enumerate(rows, 1):
            epoch = (step - 1) // 64
            ids = row["episode_ids"]
            require(type(row["step"]) is int and type(row["data_epoch"]) is int and
                    row["data_epoch"] == epoch and len(ids) == len(set(ids)) == 8 and
                    set(ids) <= fit and row["rollouts_per_episode"] == 4,
                    "Invalid compact per-step membership")
            epochs[epoch].update({episode: row["rollouts_per_episode"] for episode in ids})
        require(all(set(counts) == fit and set(counts.values()) == {4} for counts in epochs),
                "Compact epoch coverage differs")
        trajectories = sum(sum(counts.values()) for counts in epochs)
        exposures = sum(len(row["episode_ids"]) for row in rows)
        require((d["optimizer_steps"], d["data_epochs"], d["fit_unique_episodes"],
                 d["groups_per_step"], d["group_size"], d["episode_exposures"],
                 d["recorded_rollout_trajectories"]) ==
                (128, 2, 512, 8, 4, exposures, trajectories), "Reported budget totals disagree")
        membership_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        require(d["batch_membership_sha256"] == membership_sha, "Membership fingerprint differs")
        batches_by_arm_seed[arm, seed] = rows
        checked.append({"run": label, "export_sha256": digest(path),
                        "recorded_rollout_sha256": d["rollout_sha256"],
                        "episode_exposures": exposures, "recorded_trajectories": trajectories,
                        "batch_membership_sha256": membership_sha})
    paired_seeds = []
    for seed in (11, 22, 33):
        if ("control", seed) in batches_by_arm_seed and ("candidate", seed) in batches_by_arm_seed:
            require(batches_by_arm_seed["control", seed] == batches_by_arm_seed["candidate", seed],
                    "Matched arms use different per-step episode members")
            paired_seeds.append(seed)
    report = {"schema": "positive_trajectory_compact_rollout_budget_recount_v1",
              "fit_manifest_sha256": digest(manifest_path), "raw_auditor_sha256": source_sha,
              "compact_verifier_sha256": digest(Path(__file__)), "arms_checked": checked,
              "matched_step_membership_seeds_checked": paired_seeds,
              "scope": "Independent tally of compact training membership and budget, linked to copied audit/scalar provenance; not an independent raw-trajectory or physical-scene check",
              "navigation_results_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
