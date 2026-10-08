#!/usr/bin/env python3
"""Recount recorded rollout membership and budget for a completed scale arm."""

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    label = "positive_trajectory_{}_128step_seed{}".format(args.arm, args.seed)
    run = args.root / "runlogs" / label
    require((run / "completed").is_file() and not (run / "failed").exists(),
            "Training has not successfully completed")
    completed = (run / "completed").read_text().strip()
    require(bool(completed), "Empty completion marker")
    manifest_path = args.root / "prepared_data/fit512_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = manifest["episode_ids"]
    require(len(ids) == len(set(ids)) == 512 and all(type(x) is str for x in ids),
            "Frozen fit episode set differs")
    fit = set(ids)
    parquet_sha = sha256(args.root / "prepared_data/fit512.parquet")
    require(parquet_sha == manifest["generated_parquet_sha256"] ==
            "240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e",
            "Frozen fit parquet identity differs")
    config_path = run / "config.txt"
    parts = [item.split("=", 1) for item in config_path.read_text().split()]
    config = dict(parts)
    require(len(config) == len(parts), "Duplicate configuration key")
    fixed = {"arm": args.arm, "seed": str(args.seed), "steps": "128",
             "rows": "512", "batch_rows": "8", "group_n": "4",
             "dataset_sha256": parquet_sha}
    require(all(config.get(key) == value for key, value in fixed.items()),
            "Frozen training budget or run identity differs")
    audit_path = args.root / "runlogs/positive_scale" / (
        "{}_seed{}_train_audit.json".format(args.arm, args.seed))
    audit = json.loads(audit_path.read_text())
    require(audit["arm"] == args.arm and
            audit["expected_steps"] == audit["observed_steps"] == 128 and
            audit["schema"] == "positive_trajectory_training_audit_v1",
            "Optimizer audit identity or coverage differs")
    log_path = run / "train.log"
    require(audit["train_log_sha256"] == sha256(log_path), "Audited log changed")
    logged_epochs = {}
    for line in log_path.open(errors="replace"):
        hit = re.search(r"\bstep:(\d+) - global_seqlen", line)
        if hit:
            step = int(hit.group(1))
            epoch = re.search(r"training/epoch:([-+0-9.eE]+)", line)
            require(step not in logged_epochs and epoch is not None,
                    "Missing or duplicate logged epoch")
            logged_epochs[step] = float(epoch.group(1))
    require(sorted(logged_epochs) == list(range(1, 129)), "Logged step coverage differs")

    rollout_path = args.root / "verl_checkpoints" / label / "rollout.jsonl"
    rows = []
    epoch_counts = [Counter(), Counter()]
    for line in rollout_path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        step = record["step"]
        require(type(step) is int and step == len(rows) + 1 and step <= 128,
                "Rollout step order or coverage differs")
        epoch = (step - 1) // 64
        require(logged_epochs[step] == epoch, "Rollout block and logged epoch disagree")
        info = record["info"]
        require(len(info) == 32 and all(item["data_source"] == "r2r" for item in info),
                "Per-step rollout count or data source differs")
        counts = Counter(str(item["episode_id"]) for item in info)
        require(len(counts) == 8 and set(counts.values()) == {4},
                "Per-step episode groups or group size differs")
        require(set(counts) <= fit, "Non-fit episode appears in rollout records")
        epoch_counts[epoch].update(counts)
        rows.append({"step": step, "data_epoch": epoch,
                     "episode_ids": sorted(counts), "rollouts_per_episode": 4})
    require(len(rows) == 128, "Rollout record coverage differs")
    require(all(set(counts) == fit and set(counts.values()) == {4}
                for counts in epoch_counts), "An epoch omits or repeats fit episodes")
    require((run / "completed").read_text().strip() == completed and
            not (run / "failed").exists(), "Completion state changed during recount")
    report = {
        "schema": "positive_trajectory_recorded_rollout_budget_v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run": label, "arm": args.arm, "seed": args.seed, "completed": completed,
        "optimizer_steps": 128, "data_epochs": 2, "fit_unique_episodes": 512,
        "groups_per_step": 8, "group_size": 4, "episode_exposures": 1024,
        "recorded_rollout_trajectories": sum(sum(c.values()) for c in epoch_counts),
        "fit_manifest_sha256": sha256(manifest_path), "fit_parquet_sha256": parquet_sha,
        "training_audit_sha256": sha256(audit_path), "train_log_sha256": audit["train_log_sha256"],
        "training_config_sha256": sha256(config_path),
        "rollout_sha256": sha256(rollout_path), "auditor_sha256": sha256(Path(__file__)),
        "batch_membership_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "scope": "Recorded training episode-ID membership and rollout budget; not navigation evaluation, semantic accuracy, or independent physical-scene verification",
        "steps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "run", "fit_unique_episodes", "episode_exposures", "recorded_rollout_trajectories",
        "batch_membership_sha256", "scope")}))


if __name__ == "__main__":
    main()
