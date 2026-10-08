#!/usr/bin/env python3
"""Archive CPU evidence after completed, audited positive-trajectory runs."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006")
JOBS = {"control22": ("control", 22), "candidate22": ("candidate", 22),
        "control33": ("control", 33), "candidate33": ("candidate", 33)}
HELPERS = {
    "tensorboard": ("export_positive_tensorboard.py", "133c76b36d58bcfaa13a27ca8e87e8aa90716b7b7e2592cf64fa37af84426a26", "positive_trajectory_tensorboard_training_export_v1"),
    "checkpoint_metadata": ("export_positive_checkpoint_metadata.py", "22f9b812d353e51ba355bab6479282b8fdaf7648eec6b1ac2ff775395c064684", "trained_checkpoint_metadata_extent_check_v1"),
    "rollout_budget": ("audit_positive_rollout_budget.py", "069a6b2d09fada8ff3580a8784e5d3f5f15f545999aa76bfb29e109c37bef02f", "positive_trajectory_recorded_rollout_budget_v1"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_status(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(temporary), str(path))


def owner_alive(root):
    path = root / "runlogs/positive_scale/watcher.launcher.pid"
    if not path.is_file():
        return False
    pid = int(path.read_text().strip())
    process = Path("/proc") / str(pid)
    try:
        command = (process / "cmdline").read_bytes().replace(b"\x00", b" ")
        return (process / "cwd").resolve() == root and b"run_positive_scale_if_pass.sh" in command
    except FileNotFoundError:
        return False


def validate_reports(reports, root, label, arm, seed, audit_path):
    audit_sha = sha(audit_path)
    run = root / "runlogs" / label
    log_sha, config_sha = sha(run / "train.log"), sha(run / "config.txt")
    for key, (_, expected_source, schema) in HELPERS.items():
        report = reports[key]
        require(report["schema"] == schema, "Evidence schema differs")
        require(report.get("run", report.get("label")) == label, "Evidence run differs")
        require(report["training_audit_sha256"] == audit_sha and
                report["train_log_sha256"] == log_sha, "Evidence audit/log identities differ")
        require(report.get("training_config_sha256", report.get("config_sha256")) == config_sha,
                "Evidence training configuration differs")
        require(report.get("exporter_sha256", report.get("auditor_sha256")) == expected_source,
                "Evidence helper source differs")
    tb, metadata, budget = (reports[key] for key in HELPERS)
    require(tb["arm"] == arm and tb["seed"] == seed and tb["expected_steps"] == 128,
            "Scalar export identity differs")
    require(metadata["tensors"] == 825 and len(metadata["shards"]) == 4,
            "Checkpoint structure coverage differs")
    require(budget["arm"] == arm and budget["seed"] == seed and
            budget["optimizer_steps"] == 128 and budget["data_epochs"] == 2 and
            budget["group_size"] == 4 and budget["recorded_rollout_trajectories"] == 4096,
            "Recorded rollout budget differs")


def collect(root, archive, python, job):
    arm, seed = JOBS[job]
    label = "positive_trajectory_{}_128step_seed{}".format(arm, seed)
    run = root / "runlogs" / label
    require(not (run / "failed").exists(), "Training failed: " + label)
    audit_path = root / "runlogs/positive_scale" / ("{}_seed{}_train_audit.json".format(arm, seed))
    if not (run / "completed").is_file() or not audit_path.is_file():
        return {"run": label, "state": "waiting-for-completed-training-and-audit"}
    try:
        audit = json.loads(audit_path.read_text())
    except json.JSONDecodeError:
        return {"run": label, "state": "waiting-for-audit-write"}
    require(audit["schema"] == "positive_trajectory_training_audit_v1" and
            audit["arm"] == arm and audit["expected_steps"] == audit["observed_steps"] == 128,
            "Completed training audit differs")
    outputs, reports = {}, {}
    for key, (helper, _, _) in HELPERS.items():
        output = archive / ("{}_seed{}_{}.json".format(arm, seed, key))
        if not output.exists():
            temporary = output.with_name(output.name + ".building." + uuid.uuid4().hex)
            log = archive / (temporary.name + ".log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            command = [str(python), str(root / "tools" / helper), "--root", str(root),
                       "--arm", arm, "--seed", str(seed), "--output", str(temporary)]
            with log.open("x") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            require(result.returncode == 0, "CPU helper failed; preserved log: " + str(log))
            require(not output.exists(), "Evidence output appeared during collection")
            # Link refuses an existing destination, even in a concurrent writer race.
            os.link(str(temporary), str(output))
            temporary.unlink()
        reports[key] = json.loads(output.read_text())
        outputs[key] = {"path": str(output.relative_to(archive)), "sha256": sha(output)}
    validate_reports(reports, root, label, arm, seed, audit_path)
    require((run / "completed").is_file() and not (run / "failed").exists(),
            "Training completion state changed during collection")
    return {"run": label, "state": "archived", "training_audit_sha256": sha(audit_path),
            "reports": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", choices=sorted(JOBS),
                        default=["candidate22", "control33", "candidate33"])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root, archive = args.root.resolve(), args.archive.resolve()
    require(root == ROOT, "Collector is restricted to the frozen positive source tree")
    require(len(args.jobs) == len(set(args.jobs)), "Duplicate reporting job")
    python = root.parent / "activevln_train_env/bin/python"
    require(python.is_file(), "CPU export runtime unavailable")
    archive.mkdir(parents=True, exist_ok=True)
    with (archive / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        try:
            while True:
                for helper, expected, _ in HELPERS.values():
                    require(sha(root / "tools" / helper) == expected, "Frozen CPU helper source changed")
                jobs = {job: collect(root, archive, python, job) for job in args.jobs}
                done = all(item["state"] == "archived" for item in jobs.values())
                status = {"schema": "positive_cpu_evidence_collector_v1",
                          "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "pid": os.getpid(), "collector_sha256": sha(Path(__file__)),
                          "helper_sha256": {name: value for name, value, _ in HELPERS.values()},
                          "jobs": jobs, "all_requested_evidence_archived": done,
                          "scope": "CPU training-evidence collection only; no GPU work, navigation result, tensor-value validation, or changes to training/evaluation markers"}
                write_status(archive / "status.json", status)
                current = json.dumps(jobs, sort_keys=True)
                if current != previous:
                    print(json.dumps(status), flush=True)
                    previous = current
                if done or args.once:
                    return
                require(not (root / "runlogs/positive_scale/suite.failed").exists(),
                        "Primary scale suite failed while evidence remains pending")
                require(owner_alive(root), "Primary scale owner is not alive while evidence remains pending")
                time.sleep(30)
        except Exception as error:
            write_status(archive / "failure.json", {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                    "pid": os.getpid(), "error": str(error),
                                                    "scope": "Evidence collector failure; primary training/evaluation markers untouched"})
            raise


if __name__ == "__main__":
    main()
